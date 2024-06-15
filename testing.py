from multiprocessing import Pool
from transparency_selection import double_dip_transparency, TransparencyConfig
from segmentation import double_dip_segmentation, SegmentationConfig
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import io, transforms
import numpy as np
import argparse
import os

from dip import DIP
from utils import save


image_path = "curry-balloon.jpeg"

def run_transparency(x, img):
    config = TransparencyConfig(img, f"output_transparency/im_{x}.png", 512, 5000, 1, 0.1, False)
    return double_dip_transparency(config)

def run_segmentation(x, img):
    config = SegmentationConfig(img, f"output_segmentation/im_{x}.png", 512, 1000, 5, 1000000, 0.1, False)
    return double_dip_segmentation(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run double DIP")
    parser.add_argument('image', type=str)
    parser.add_argument('type', choices=["segmentation", "transparency"])
    parser.add_argument('--num', type=int, default=10)

    args = parser.parse_args()

    with Pool() as p:
        if args.type == "segmentation":
            os.makedirs(f"./output_segmentation/", exist_ok=True)
            p.starmap(run_segmentation, zip(range(args.num), [args.image]*args.num))
        elif args.type == "transparency":
            os.makedirs(f"./output_transparency/", exist_ok=True)
            p.starmap(run_transparency, zip(range(args.num), [args.image]*args.num))
