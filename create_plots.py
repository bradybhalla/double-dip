# run after generate-results

import matplotlib.pyplot as plt
from PIL import Image
import os

def clear_axis(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

os.makedirs("plots", exist_ok=True)

# denoising
noises = ["15", "30", "50", "80"]
best_epochs = [5000, 3500, 2000, 1500]

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
fig.tight_layout()
axs[0][0].set_ylabel("Input")
axs[1][0].set_ylabel("Denoised")

for i, (n,b) in enumerate(zip(noises, best_epochs)):
    with Image.open(f"output/denoising-{n}/input.png") as im:
        clear_axis(axs[0][i])
        axs[0][i].set_title(f"$\\sigma = 0.{n}$")
        axs[0][i].imshow(im)

    with Image.open(f"output/denoising-{n}/epoch-{b:06}.png") as im:
        clear_axis(axs[1][i])
        axs[1][i].imshow(im)

plt.subplots_adjust(wspace=0.01, hspace=-0.49)
plt.savefig("plots/denoising.png", bbox_inches="tight", pad_inches=0.01, transparent="True", dpi=500)

# inpainting
percents = ["15", "30", "50", "80"]
best_epochs = [5000, 5000, 5000, 5000]

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
fig.tight_layout()
axs[0][0].set_ylabel("Input")
axs[1][0].set_ylabel("Inpainted")

for i, (n,b) in enumerate(zip(percents, best_epochs)):
    with Image.open(f"output/inpainting-{n}/input.png") as im:
        clear_axis(axs[0][i])
        axs[0][i].set_title(f"{n}% Removed")
        axs[0][i].imshow(im)

    with Image.open(f"output/inpainting-{n}/epoch-{b:06}.png") as im:
        clear_axis(axs[1][i])
        axs[1][i].imshow(im)

plt.subplots_adjust(wspace=0.01, hspace=-0.49)
plt.savefig("plots/inpainting.png", bbox_inches="tight", pad_inches=0.01, transparent="True", dpi=500)

# segmentation
best_epochs = [1000, 1000, 1000, 1000]

fig, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(6.4,6.45))
fig.tight_layout()
axs[0][0].set_title("Input")
axs[0][1].set_title("Mask")
axs[0][2].set_title("Component 1")
axs[0][3].set_title("Component 2")

for i, b in enumerate(best_epochs):
    for j,title in enumerate(["original", f"epoch-{b:06}-mask", f"epoch-{b:06}-im1", f"epoch-{b:06}-im2"]):
        with Image.open(f"output/segmentation-{i}/{title}.png") as im:
            clear_axis(axs[i][j])
            axs[i][j].imshow(im)

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig("plots/segmentation.png", bbox_inches="tight", pad_inches=0.01, transparent="True", dpi=500)

# transparency separation
info = [(5000, "img40", "img70"), (5000, "img40", "img75"), (5000, "img40", None)]

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
fig.tight_layout()
axs[0][0].set_title("Input 1")
axs[0][1].set_title("Input 2")
axs[0][2].set_title("Component 1")
axs[0][3].set_title("Component 2")

for i, (b, t1, t2) in enumerate(info):
    for j, title in enumerate([f"input-{t1}", f"input-{t2}", f"epoch-{b:06}-im1", f"epoch-{b:06}-im2"]):
        if title == "input-None":
            axs[i][j].set_axis_off()
            continue
        with Image.open(f"output/transparent-{i}/{title}.png") as im:
            clear_axis(axs[i][j])
            axs[i][j].imshow(im)

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig("plots/transparency-separation.png", bbox_inches="tight", pad_inches=0.01, transparent="True", dpi=500)

# watermark removal
info = [
    (5000, ["img1", "img2", "img3"]),
    (5000, ["img1", "img2", "img3"]),
    (5000, ["img1", "img2", "img3"]),
    (5000, ["img1", "img2", "img3"]),
]

for i, (b, images) in enumerate(info):
    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
    fig.tight_layout()
    axs[0][0].set_ylabel("Input")
    axs[1][0].set_ylabel("Output")

    for j, title in enumerate(images):
        with Image.open(f"output/watermark-{i}/input-{title}.png") as im:
            clear_axis(axs[0][j])
            axs[0][j].imshow(im)
        with Image.open(f"output/watermark-{i}/epoch-{b:06}-reconstructed-{title}.png") as im:
            clear_axis(axs[1][j])
            axs[1][j].imshow(im)

    axs[0][3].set_axis_off()
    with Image.open(f"output/watermark-{i}/epoch-{b:06}-watermark.png") as im:
        clear_axis(axs[1][3])
        axs[1][3].imshow(im)

    plt.subplots_adjust(wspace=0.01, hspace=-0.49)
    plt.savefig(f"plots/watermark-{i}.png", bbox_inches="tight", pad_inches=0.01, transparent="True", dpi=500)
