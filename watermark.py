from dataclasses import dataclass
import itertools
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import io, transforms
import numpy as np
import argparse
import torch.nn as nn
import glob

from dip import DIP
from utils import clear_dir, save


device = torch.device("cuda")

@dataclass
class WatermarkConfig:
    image_paths: list[str]
    save_dir: str
    img_size: int
    epochs: int
    excl_coeff: float
    mask_coeff: float
    lr: float
    verbose: bool


class DoubleDIPWatermark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        dip_args = {
            "z_channels": 32,
            "z_scale": 1/10,
            "z_noise": 0,
            "filters_down" : [16, 32, 64, 128, 128, 128],
            "filters_up" : [16, 32, 64, 128, 128, 128],
            "kernels_down" : [3, 3, 3, 3, 3, 3],
            "kernels_up" : [5, 5, 5, 5, 5, 5],
            "filters_skip" : [4, 4, 4, 4, 4, 4],
            "kernels_skip" : [1, 1, 1, 1, 1, 1],
            "upsampling" : "nearest"
        }
        
        self.images = nn.ModuleList([DIP(config.img_size, 3, **dip_args).to(device) for _ in config.image_paths])
        self.watermark = nn.Parameter(torch.tensor(1.0, device=device))
        self.mask = DIP(config.img_size, 1, **dip_args).to(device)

    def forward(self):
        orig_images = []
        reconstructed = []
        mask = self.mask()
        watermark = F.sigmoid(self.watermark)
        for i in self.images:
            im = i()
            orig_images.append(im)
            reconstructed.append(im*mask + 1 - mask)
        return orig_images, watermark, mask, reconstructed

def double_dip_watermark(config: WatermarkConfig):
    clear_dir(config.save_dir)

    targets = []
    for i in config.image_paths:
        transform = transforms.Resize((config.img_size, config.img_size))
        im = io.read_image(i) / 255
        targets.append(transform(im).clip(0,1).unsqueeze(0).to(device))
        save(targets[-1], f"{config.save_dir}/input-{i.split('/')[-1]}")

    double_dip = DoubleDIPWatermark(config)

    optimizer = torch.optim.Adam(double_dip.parameters(), lr=config.lr)

    double_dip.train()

    save_points = set([i for i in np.arange(0, config.epochs, (config.epochs+1)//10)] + [config.epochs])
    saves = []

    if config.verbose:
        rg = tqdm(range(config.epochs+1))
    else:
        rg = range(config.epochs+1)
    for i in rg:

        orig_images, watermark, mask, reconstructed = double_dip()

        loss_rec = 0
        for j, y in enumerate(reconstructed):
            loss_rec += torch.norm(y - targets[j])

        loss_mask = 1/torch.sum(torch.abs(mask - 0.5))

        loss_excl = 0
        for im1, im2 in itertools.combinations(orig_images, 2):
            downsample_im1 = im1
            downsample_im2 = im2
            for n in range(3):
                im1_grad = torch.cat(torch.gradient(downsample_im1[0], dim=(1,2)))
                im2_grad = torch.cat(torch.gradient(downsample_im2[0], dim=(1,2)))
                lambda_1 = torch.sqrt(torch.norm(im2_grad) / torch.norm(im1_grad))
                lambda_2 = torch.sqrt(torch.norm(im1_grad) / torch.norm(im2_grad))
                loss_excl += torch.norm(torch.tanh(lambda_1 * torch.abs(im1_grad)) * torch.tanh(lambda_2 * torch.abs(im2_grad)))

                downsample_im1 = F.interpolate(downsample_im1, scale_factor=1/2, mode="bilinear")
                downsample_im2 = F.interpolate(downsample_im2, scale_factor=1/2, mode="bilinear")

        loss = loss_rec + config.excl_coeff * loss_excl + config.mask_coeff * loss_mask

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.verbose:
            rg.set_description(f"(Loss {loss.item():.4f}, alpha {watermark.item():.4f})")

        if i in save_points:
            saves.append(torch.cat((*orig_images, (1 - mask.repeat(1,3,1,1))), dim=3))
            save(torch.cat(saves, dim=2), f"{config.save_dir}/saves.png")
            save(saves[-1], f"{config.save_dir}/epoch-{i:06}.png")
            for j,o in zip(config.image_paths, orig_images):
                save(o, f"{config.save_dir}/epoch-{i:06}-reconstructed-{j.split('/')[-1]}")
            save((1 - mask.repeat(1,3,1,1)), f"{config.save_dir}/epoch-{i:06}-watermark.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double DIP watermark removal")
    parser.add_argument('image_dir', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--excl_coeff', type=float, default=5)
    parser.add_argument('--mask_coeff', type=float, default=10000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    image_paths = []
    for i in glob.glob(f"{args.image_dir}/*"):
        image_paths.append(i)


    config = WatermarkConfig(
        image_paths = image_paths,
        save_dir = args.save_dir,
        img_size = args.img_size,
        epochs = args.epochs,
        excl_coeff = args.excl_coeff,
        mask_coeff = args.mask_coeff,
        lr = args.lr,
        verbose = args.verbose,
    )
    
    double_dip_watermark(config)
