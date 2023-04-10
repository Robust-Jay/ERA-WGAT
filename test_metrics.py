import argparse
import json
from skimage.metrics import *
import torch
import numpy as np
import torch.nn.functional as F
import math
from tqdm import tqdm
from model import ERA_WGAT
from torchvision.models import vgg19


def grey_to_rgb(data):
    data = torch.from_numpy(data).unsqueeze(0)
    return torch.cat([data for _ in range(3)], dim=0)


def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    args.eval_metrics = True
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    args.config = config
    args.device = torch.device('cuda:1')

    main_worker(args)


def main_worker(args):
    device = args.device
    model = ERA_WGAT(args.config)

    # kwargs = {'map_location': lambda storage, loc: storage.cuda(0)}
    # model = load_GPUS(model, f"{args.config['out']}/best.pkl", kwargs)

    state_dict = torch.load(f"{args.config['out']}/60.pkl", map_location=lambda storage, loc: storage.cuda(1))
    model.load_state_dict(state_dict)

    model.to(device)

    eval(model, device)


def eval(model, device):
    vgg_model = vgg19(pretrained=True).features[:35].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    root = "../data/Mayo"
    test_files = np.load('dataset/split_datasets/test.npy', allow_pickle=False)

    RMSE = []
    SSIM = []
    PSNR = []
    VGGP = []

    with torch.no_grad():
        model.eval()
        for file in tqdm(test_files):
            y_t = np.load(f"{root}/full_dose/{file}")
            noise = np.load(f"{root}/quarter_dose/{file}")
            noise = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0).to(device)

            y_hat = model(noise)
            y_p = y_hat.detach().cpu().squeeze(0).squeeze(0).numpy()
            mse = mean_squared_error(y_t, y_p)
            rmse = math.sqrt(mse)
            data_range = 1
            ssim = structural_similarity(y_t, y_p, data_range=data_range)
            psnr = peak_signal_noise_ratio(y_t, y_p, data_range=data_range)

            RMSE.append(rmse)
            SSIM.append(ssim)
            PSNR.append(psnr)

            with torch.no_grad():
                y_t = grey_to_rgb(y_t).to(device).unsqueeze(0)
                y_p = grey_to_rgb(y_p).to(device).unsqueeze(0)

                y_t_vgg = vgg_model(y_t)
                y_p_vgg = vgg_model(y_p)

                vgg_p = F.mse_loss(y_p_vgg, y_t_vgg)

                VGGP.append(vgg_p.cpu().numpy())

    print(f"Mean \t RMSE: {np.mean(RMSE)}\t PSNR: {np.mean(PSNR)}\t SSIM: {np.mean(SSIM)}\t VGGP: {np.mean(VGGP)}")
    print(f"Std \t RMSE: {np.std(RMSE)}\t PSNR: {np.std(PSNR)}\t SSIM: {np.std(SSIM)}\t VGGP: {np.std(VGGP)}")


if __name__ == '__main__':
    main()
