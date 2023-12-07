import torch
import matplotlib.pyplot as plt
import os

from torchvision.io import read_image, write_png
from torchvision.transforms import Resize, CenterCrop

def style_and_ref():
    path_style = os.path.join("mic_ablations_style", "1")
    path_ref = os.path.join("data", "mic", "val", "r_2.png")

    style = read_image(os.path.join(path_style, "style_image.png"))
    ref = read_image(path_ref)

    img = ref.clone()[:3]
    s = 300
    t = Resize(size=(s, s))
    reshape_style = t(style)

    img[:, img.shape[1] - s-2:, img.shape[-1] - s-2:] = torch.tensor((255, 0, 0))[:, None, None]
    img[:, img.shape[1] - s:, img.shape[-1] - s:] = reshape_style

    plot, ax = plt.subplots(1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img.permute(1, 2, 0))

    ax.text(x=600, y=450, s='Style Image', color='white')

    plt.savefig(os.path.join(path_style, "style_and_ref.png"), bbox_inches='tight', pad_inches=0.)
    plt.close()

if __name__ == '__main__':
    style_and_ref()