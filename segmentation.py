from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import io, transforms
import numpy as np
import argparse

from dip import DIP
from utils import save, clear_dir

@dataclass
class SegmentationConfig:
    image_path: str
    save_dir: str
    img_size: int
    epochs: int
    excl_coeff: float
    mask_coeff: float
    lr: float
    verbose: bool

def double_dip_segmentation(config: SegmentationConfig):
    clear_dir(config.save_dir)

    device = torch.device("cuda")

    transform = transforms.Resize((config.img_size, config.img_size))
    im = io.read_image(config.image_path) / 255
    target = transform(im).clip(0,1).unsqueeze(0).to(device)
    save(target, f"{config.save_dir}/original.png")

    dip_args = {
        "z_channels": 32,
        "z_scale": 1,
        "z_noise": 1/10,
        "filters_down" : [16, 32, 64, 128, 128, 128],
        "filters_up" : [16, 32, 64, 128, 128, 128],
        "kernels_down" : [3, 3, 3, 3, 3, 3],
        "kernels_up" : [5, 5, 5, 5, 5, 5],
        "filters_skip" : [0, 0, 1, 0, 1, 0],
        "kernels_skip" : [1, 1, 1, 1, 1, 1],
        "upsampling" : "nearest"
    }

    dip0 = DIP(config.img_size, 1, **dip_args).to(device)
    dip1 = DIP(config.img_size, 3, **dip_args).to(device)
    dip2 = DIP(config.img_size, 3, **dip_args).to(device)

    optimizer = [torch.optim.Adam(i.parameters(), lr=config.lr) for i in [dip0, dip1, dip2]]

    dip0.train()
    dip1.train()
    dip2.train()

    save_points = set([i for i in np.arange(0, config.epochs, (config.epochs+1)//10)] + [config.epochs])
    saves = []

    if config.verbose:
        rg = tqdm(range(config.epochs+1))
    else:
        rg = range(config.epochs+1)
    for i in rg:
        mask = dip0()
        im1 = dip1()
        im2 = dip2()

        loss_excl = 0
        downsample_im1 = im1
        downsample_im2 = im2
        for n in range(8):
            im1_grad = torch.cat(torch.gradient(downsample_im1[0], dim=(1,2)))
            im2_grad = torch.cat(torch.gradient(downsample_im2[0], dim=(1,2)))
            lambda_1 = torch.sqrt(torch.norm(im2_grad) / torch.norm(im1_grad))
            lambda_2 = torch.sqrt(torch.norm(im1_grad) / torch.norm(im2_grad))
            loss_excl += torch.norm(torch.tanh(lambda_1 * torch.abs(im1_grad)) * torch.tanh(lambda_2 * torch.abs(im2_grad)))

            downsample_im1 = F.interpolate(downsample_im1, scale_factor=1/2, mode="bilinear")
            downsample_im2 = F.interpolate(downsample_im2, scale_factor=1/2, mode="bilinear")

        y = im1 * mask + im2 * (1 - mask)

        loss = torch.norm(y - target) + config.excl_coeff * loss_excl + config.mask_coeff * 1/torch.sum(torch.abs(mask - 0.5))

        for o in optimizer:
            o.zero_grad()

        loss.backward()

        for o in optimizer:
            o.step()

        if config.verbose:
            rg.set_description(f"(Loss {loss.item():.4f})")

        if i in save_points:
            saves.append(torch.cat((im1, im2, mask.repeat(1,3,1,1), y), dim=3))
            save(torch.cat(saves, dim=2), f"{config.save_dir}/saves.png")
            save(saves[-1], f"{config.save_dir}/epoch-{i:06}.png")
            save(im1, f"{config.save_dir}/epoch-{i:06}-im1.png")
            save(im2, f"{config.save_dir}/epoch-{i:06}-im2.png")
            save(mask.repeat(1,3,1,1), f"{config.save_dir}/epoch-{i:06}-mask.png")
            save(y, f"{config.save_dir}/epoch-{i:06}-reconstructed.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double DIP segmentation")
    parser.add_argument('image_path', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--excl_coeff', type=float, default=5)
    parser.add_argument('--mask_coeff', type=float, default=1000000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    config = SegmentationConfig(
        image_path = args.image_path,
        save_dir = args.save_dir,
        img_size = args.img_size,
        epochs = args.epochs,
        excl_coeff = args.excl_coeff,
        mask_coeff = args.mask_coeff,
        lr = args.lr,
        verbose = args.verbose,
    )
    
    double_dip_segmentation(config)
