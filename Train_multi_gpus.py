import os
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import random
from tqdm import trange

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import CTDataset
from model import ERA_WGAT
from metrics import Torch_MSE, Torch_PSNR, Torch_SSIM, VGGP
from compound_loss import CompoundLoss
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a configs.json file with training/model/data/param details")
    parser.add_argument('--local_rank', type=int)

    # local_rank = int(os.environ["LOCAL_RANK"])

    args = parser.parse_args()
    args.nprocs = 2
    print(args.config)
    with open(args.config) as f:
        config = json.load(f)

    args.config = config

    main_worker(args.local_rank, args)


def main_worker(local_rank, args):
    seed = args.config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    best_val_loss = 10000

    dist.init_process_group(backend='nccl', rank=local_rank)  #  ubuntu
    # dist.init_process_group(backend='gloo', rank=local_rank)  #  windows
    torch.cuda.set_device(local_rank)

    model = ERA_WGAT(args.config)
    # state_dict = torch.load("out/loss/MSE_VGG/0.0001/45.pkl", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if args.config['loss'] == 'MSE':
        criterion = nn.MSELoss().cuda(local_rank)
    else:
        criterion = CompoundLoss(args.config['VGG_ratio']).cuda(local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=args.config['init_lr'], weight_decay=args.config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # for i in range(45):
    #     scheduler.step()
    
    cudnn.benchmark = True

    trainset = CTDataset(flag='train')
    valset = CTDataset(flag='val')
    testset = CTDataset(flag='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    batch_size = args.config['batch_size'] // args.nprocs

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True,
                              sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
                            sampler=val_sampler)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True,
                             sampler=test_sampler)

    results = pd.DataFrame(columns=('train_loss',
                                    'val_loss', 'val_RMSE', 'val_PSNR', 'val_SSIM', 'val_VGGP',
                                    'test_loss', 'test_RMSE', 'test_PSNR', 'test_SSIM', 'test_VGGP'))

    for epoch in range(args.config['epochs']):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, criterion, optimizer, local_rank, args)
        val_loss, val_RMSE, val_PSNR, val_SSIM, val_VGGP = validate(val_loader, model, criterion, local_rank, args)
        test_loss, test_RMSE, test_PSNR, test_SSIM, test_VGGP = validate(test_loader, model, criterion, local_rank, args)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if local_rank == 0:
            results = results.append(pd.DataFrame({'train_loss': [train_loss],
                                                   'val_loss': [val_loss],
                                                   'val_RMSE': [val_RMSE],
                                                   'val_PSNR': [val_PSNR],
                                                   'val_SSIM': [val_SSIM],
                                                   'val_VGGP': [val_VGGP],
                                                   'test_loss': [test_loss],
                                                   'test_RMSE': [test_RMSE],
                                                   'test_PSNR': [test_PSNR],
                                                   'test_SSIM': [test_SSIM],
                                                   'test_VGGP': [test_VGGP]}), ignore_index=True)

            print(f"Epoch {epoch + 1}\t lr: {optimizer.param_groups[0]['lr']} "
                  f"\ttrain_loss: {train_loss}\n"
                  f"\t val_RMSE: {val_RMSE} \t val_PSNR: {val_PSNR} \t val_SSIM: {val_SSIM} \t val_VGGP: {val_VGGP}\n"  # noqa
                  f"\ttest_RMSE: {test_RMSE} \ttest_PSNR: {test_PSNR} \ttest_SSIM: {test_SSIM} \ttestVGGP: {test_VGGP}")  # noqa

            save_checkpoint(epoch, results, model.module.state_dict(), args.config['out'], is_best)  # noqa
        scheduler.step()


def train(train_loader, model, criterion, optimizer, local_rank, args):
    model.train()

    loss_meter = AverageMeter()
    scaler = GradScaler()

    with trange(len(train_loader)) as t:
        for iter, (input_images, gt_images) in zip(t, train_loader):
            input_images = input_images.cuda(local_rank, non_blocking=True)
            gt_images = gt_images.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                pred = model(input_images)
                loss = criterion(pred, gt_images)
            # pred = model(input_images)
            # loss = criterion(pred, gt_images)
            dist.barrier()
            # print(input_images.size(0))
            reduced_loss = reduce_mean(loss, args.nprocs)
            loss_meter.update(reduced_loss.item())
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t.set_postfix(iter=iter + 1, loss_img=loss_meter.avg)

    return loss_meter.avg


def validate(val_loader, model, criterion, local_rank, args):
    model.eval()

    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    vggp_meter = AverageMeter()

    with torch.no_grad():
        for iter, (input_images, gt_images) in enumerate(val_loader):
            input_images = input_images.cuda(local_rank, non_blocking=True)
            gt_images = gt_images.cuda(local_rank, non_blocking=True)
            pred = model(input_images)
            loss = criterion(pred, gt_images)
            rmse, psnr, ssim, vggp = analysis_metrics(pred, gt_images)
            dist.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_rmse = reduce_mean(rmse, args.nprocs)
            reduced_psnr = reduce_mean(psnr, args.nprocs)
            reduced_ssim = reduce_mean(ssim, args.nprocs)
            reduced_vggp = reduce_mean(vggp, args.nprocs)

            loss_meter.update(reduced_loss.item())
            rmse_meter.update(reduced_rmse.item())
            psnr_meter.update(reduced_psnr.item())
            ssim_meter.update(reduced_ssim.item())
            vggp_meter.update(reduced_vggp.item())

    return loss_meter.avg, rmse_meter.avg, psnr_meter.avg, ssim_meter.avg, vggp_meter.avg


def analysis_metrics(pred, target):
    with torch.no_grad():
        metric_mse = Torch_MSE()
        metric_psnr = Torch_PSNR()
        metric_ssim = Torch_SSIM()
        metric_vggp = VGGP()

        mse = metric_mse(pred, target)
        rmse = torch.sqrt(mse)
        psnr = metric_psnr(pred, target)
        ssim = metric_ssim(pred, target)
        vggp = metric_vggp(pred, target)
    return rmse, psnr, ssim, vggp


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_checkpoint(epoch, results, state_dict, out_root, is_best):
    if not os.path.exists(out_root):
        os.makedirs(out_root)
        
    results.to_csv(f"{out_root}/results.csv")
    torch.save(state_dict, f"{out_root}/{epoch + 1}.pkl")
    if is_best:
        shutil.copyfile(f"{out_root}/{epoch + 1}.pkl", f"{out_root}/best.pkl")


if __name__ == '__main__':
    main()
    # checkpoint = torch.load("checkpoint.pth", map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint["state_dict"])
