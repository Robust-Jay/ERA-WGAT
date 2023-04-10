import os
import argparse
import json

import numpy as np

import torch
import cv2
from tqdm import tqdm
from model import ERA_WGAT


def load_GPUS(model, model_path, kwargs):
    state_dict = torch.load(model_path, **kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    args.config = config
    args.device = torch.device('cuda:0')

    main_worker(args)


def main_worker(args):
    device = args.device
    model = ERA_WGAT(args.config)

    # kwargs = {'map_location': lambda storage, loc: storage.cuda(0)}
    # model = load_GPUS(model, f"{args.config['out']}/best.pkl", kwargs)

    state_dict = torch.load(f"{args.config['out']}/54.pkl", map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(state_dict)

    model.to(device)

    eval(model, device, args.config['out'])


def eval(model, device, out):
    out_split = out.split('/')
    out_split[0] = "ERA_WGAT"
    save_root = '/'.join(out_split)

    root = "../../datasets/Mayo"
    test_files = np.load('dataset/split_datasets/test.npy', allow_pickle=False)

    with torch.no_grad():
        model.eval()
        for file in tqdm(test_files):
            patient, file_name = file.split('/')
            noise = np.load(f"{root}/quarter_dose/{file}")
            noise = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0).to(device)

            y_hat = model(noise)
            pred = y_hat.detach().cpu().squeeze(0).squeeze(0).numpy()

            if not os.path.exists(f"{save_root}/NPY/{patient}"):
                os.makedirs(f"{save_root}/NPY/{patient}")

            np.save(f"{save_root}/NPY/{file}", pred, allow_pickle=False)

            win_data = clip_rescale(pred, 850 / 3000, 1250 / 3000)
            win_dst_path = f"{save_root}/WIN/{patient}"
            if not os.path.exists(win_dst_path):
                os.makedirs(win_dst_path)
            win_dst_name = f"{save_root}/WIN/{file.split('.')[0] + '.png'}"
            cv2.imwrite(win_dst_name, win_data * 255.0)


def clip_rescale(data, vmin, vmax):
    clip_data = np.clip(data, vmin, vmax)
    clip_data = (clip_data - vmin) / (vmax - vmin)

    return clip_data


if __name__ == '__main__':
    main()

