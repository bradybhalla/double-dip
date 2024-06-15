from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import io, transforms
import numpy as np
import argparse
import torch.nn as nn
import glob
import os
import shutil
from pathlib import Path

from dip import DIP
from utils import save, clear_dir


device = torch.device("cuda")

@dataclass
class DenoisingConfig:
    image_path: str
    save_dir: str
    img_size: int
    epochs: int
    lr: float
    noise: float
    verbose: bool


def dip_infill(config: DenoisingConfig):
    clear_dir(config.save_dir)
    
    transform = transforms.Resize((config.img_size, config.img_size))
    im = io.read_image(config.image_path) / 255
    target = transform(im).clip(0,1).unsqueeze(0).to(device)
    save(target, f"{config.save_dir}/original.png")

    target = (target + torch.randn(*target.shape, device=target.device)*config.noise).clip(0,1)

    save(target, f"{config.save_dir}/input.png")

    dip_args = {
        "z_channels": 32,
        "z_scale": 1/10,
        "z_noise": 0,
        "filters_down" : [16, 32, 64, 128, 128, 128, 128],
        "filters_up" : [16, 32, 64, 128, 128, 128, 128],
        "kernels_down" : [3, 3, 3, 3, 3, 3, 3],
        "kernels_up" : [5, 5, 5, 5, 5, 5, 5],
        "filters_skip" : [0, 0, 0, 0, 0, 0, 0],
        "kernels_skip" : [0, 0, 0, 1, 1, 1, 1],
        "upsampling" : "nearest"
    }
    dip = DIP(config.img_size, 3, **dip_args).to(device)

    optimizer = torch.optim.Adam(dip.parameters(), lr=config.lr)

    dip.train()

    save_points = set([i for i in np.arange(0, config.epochs, (config.epochs+1)//10)] + [config.epochs])
    saves = []

    if config.verbose:
        rg = tqdm(range(config.epochs+1))
    else:
        rg = range(config.epochs+1)
    for i in rg:

        y = dip()


        loss = torch.norm(y - target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.verbose:
            rg.set_description(f"(Loss {loss.item():.4f})")

        if i in save_points:
            saves.append(y)
            save(torch.cat(saves, dim=2), f"{config.save_dir}/saves.png")
            save(saves[-1], f"{config.save_dir}/epoch-{i:06}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DIP infilling")
    parser.add_argument('image_path', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.4)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    config = DenoisingConfig(
        image_path = args.image_path,
        save_dir = args.save_dir.rstrip("/"),
        img_size = args.img_size,
        epochs = args.epochs,
        lr = args.lr,
        noise=args.noise,
        verbose = args.verbose,
    )
    
    dip_infill(config)
