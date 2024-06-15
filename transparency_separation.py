from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import io, transforms
import numpy as np
import argparse
import torch.nn as nn
import glob

from dip import DIP
from utils import save, clear_dir


device = torch.device("cuda")

@dataclass
class TransparencyConfig:
    image_paths: list[str]
    save_dir: str
    img_size: int
    epochs: int
    excl_coeff: float
    lr: float
    noise: float
    verbose: bool


class DoubleDIPTransparency(nn.Module):
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
            "filters_skip" : [0, 0, 0, 0, 0, 0],
            "kernels_skip" : [0, 0, 0, 0, 0, 0],
            "upsampling" : "nearest"
        }
        
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(1, device=device)) for _ in config.image_paths])
        self.dip1 = DIP(config.img_size, 3, **dip_args).to(device)
        self.dip2 = DIP(config.img_size, 3, **dip_args).to(device)

    def forward(self):
        outputs = []
        for i in self.alphas:
            im1 = self.dip1()
            im2 = self.dip2()
            alpha = F.sigmoid(i)
            outputs.append((im1, im2, im1*alpha + im2 * (1 - alpha)))
        return outputs


def double_dip_transparency(config: TransparencyConfig):
    clear_dir(config.save_dir)

    targets = []
    for i in config.image_paths:
        transform = transforms.Resize((config.img_size, config.img_size))
        im = io.read_image(i) / 255
        im += torch.randn(im.shape) * config.noise
        targets.append(transform(im).clip(0,1).unsqueeze(0).to(device))
        save(targets[-1], f"{config.save_dir}/input-{i.split('/')[-1]}")

    double_dip = DoubleDIPTransparency(config)

    optimizer = torch.optim.Adam(double_dip.parameters(), lr=config.lr)

    double_dip.train()

    save_points = set([i for i in np.arange(0, config.epochs, (config.epochs+1)//10)] + [config.epochs])
    saves = []

    if config.verbose:
        rg = tqdm(range(config.epochs+1))
    else:
        rg = range(config.epochs+1)
    for i in rg:

        outputs = double_dip()

        losses = []

        for j, (im1, im2, y) in enumerate(outputs):

            loss_excl = 0
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

            losses.append(torch.norm(y - targets[j]) + config.excl_coeff * loss_excl)

        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.verbose:
            rg.set_description(f"(Loss {loss.item():.4f})")

        if i in save_points:
            im1, im2, y = outputs[0]
            saves.append(torch.cat((im1, im2, y), dim=3))
            save(torch.cat(saves, dim=2), f"{config.save_dir}/saves.png")

            for j,o in zip(config.image_paths, outputs):
                save(o[2], f"{config.save_dir}/epoch-{i:06}-reconstructed-{j.split('/')[-1]}")

            save(im1, f"{config.save_dir}/epoch-{i:06}-im1.png")
            save(im2, f"{config.save_dir}/epoch-{i:06}-im2.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Double DIP transparency separation")
    parser.add_argument('image_dir', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--excl_coeff', type=float, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)


    args = parser.parse_args()

    image_paths = []
    for i in glob.glob(f"{args.image_dir}/*"):
        image_paths.append(i)


    config = TransparencyConfig(
        image_paths = image_paths,
        save_dir = args.save_dir,
        img_size = args.img_size,
        epochs = args.epochs,
        excl_coeff = args.excl_coeff,
        lr = args.lr,
        noise = args.noise,
        verbose = args.verbose,
    )
    
    double_dip_transparency(config)
