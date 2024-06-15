import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import io, transforms
import os

from dip import DIP
from utils import save

device = torch.device("cuda")

SIZE = 1024
EPOCHS = 4000
OUTPUT_DIR = "single_dip_out"

os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

transform = transforms.Resize((SIZE, SIZE))
im = io.read_image("./cats.jpg") / 255

target = transform(im).clip(0, 1).unsqueeze(0).to(device)
save(target, f"{OUTPUT_DIR}/target.png")

dip_args = {
    "filters_down" : [16, 32, 64, 128, 128, 128],
    "filters_up" : [16, 32, 64, 128, 128, 128],
    "kernels_down" : [7, 7, 5, 5, 3, 3],
    "kernels_up" : [7, 7, 5, 5, 3, 3],
    "filters_skip" : [4, 4, 4, 4, 4, 4],
    "kernels_skip" : [1, 1, 1, 1, 1, 1],
    "upsampling" : "nearest"
}
dip = DIP(SIZE, 3, **dip_args).to(device)

loss_fn = nn.MSELoss()
criterion = torch.optim.Adam(dip.parameters(), lr=0.01)

dip.train()

pbar = tqdm(range(EPOCHS+1))
for i in pbar:
    y = dip()
    loss = loss_fn(y, target)

    criterion.zero_grad()
    loss.backward()
    criterion.step()

    pbar.set_description(f"(Loss {loss.item():.4f})")

    if i % 50 == 0:
        save(y, f"{OUTPUT_DIR}/epoch-{i:05}.png")

