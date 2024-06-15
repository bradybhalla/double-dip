import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import shutil
import os

def save(x, file):
    arr = x.cpu().detach().numpy()[0].transpose(1,2,0)
    Image.fromarray(np.uint8(arr.clip(0,1)*255)).save(file)

def clear_dir(dir):
    if Path(dir).is_dir():
        shutil.rmtree(dir)
    os.makedirs(dir)

