import os

import torch as th
from torchvision.io import write_png

def write_as_png(tensor: th.cuda.FloatTensor, str_: str,  min: float = 0, max: float = 1):
    t = th.clamp(tensor, min, max)

    if t.shape[-1] in [1, 3] and len(t.shape) == 3:
        t = t.permute(-1, 0, 1)

    if t.requires_grad:
        t = t.detach()
    if t.is_cuda:
        t = t.cpu()

    # normalize to [0,1]
    t = (t - t.min()) / (t.max() - t.min())

    # to byte
    t = (t * 255.).byte()

    # to 3D
    if len(t.shape) == 2:
        t = t[None, ...]

    path = 'test_images'
    if not os.path.exists(path):
        os.mkdir(path)

    write_png(t, os.path.join(path, str_))

def write_palette_as_img(palet, str_):
    img = th.zeros((200, 50, 3), dtype=th.float32)
    palet = th.clamp(palet, 0, 1)
    z = 200 // palet.shape[0]
    for i in range(palet.shape[0]):
        img[i * z:(i+1)*z, :] = palet[i][None, None, ...]

    path = 'test_images'
    if not os.path.exists(path):
        os.mkdir(path)

    if img.is_cuda:
        img.cpu()

    img = (img * 255).byte()

    write_png(img.permute(-1, 0, 1), os.path.join(path, str_))