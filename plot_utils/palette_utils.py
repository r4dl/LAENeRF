import torch as th
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchvision.io import write_png

def palette_change_to_img(palet1, palet2, path, prefix: str = ''):
    size = [800, 400, 3]
    img = np.ones((size), dtype=np.float32)

    midpoints = size[0] // (palet1.shape[0] + 1)
    steps_y = np.arange(start=midpoints, stop=size[0], step=midpoints)
    size_x = steps_y[0] // 2 - 20

    midpoints = size[1] // (palet1.shape[1] + 1)
    steps_x = np.arange(start=midpoints, stop=size[1], step=midpoints * 2)
    size_x = min(size_x, steps_x[0] // 2 - 20)
    plot, ax = plt.subplots(1,1)

    for i in range(palet1.shape[0]):
        img[steps_y[i] - size_x - 2:steps_y[i] + size_x + 3,
            steps_x[0] - size_x - 2:steps_x[0] + size_x + 3] = np.zeros(3)
        img[steps_y[i] - size_x:steps_y[i] + size_x,
            steps_x[0] - size_x:steps_x[0] + size_x] = palet1[i]
        img[steps_y[i] - size_x - 2:steps_y[i] + size_x + 3,
            steps_x[1] - size_x - 2:steps_x[1] + size_x + 3] = np.zeros(3)
        img[steps_y[i] - size_x:steps_y[i] + size_x,
            steps_x[1] - size_x:steps_x[1] + size_x] = palet2[i]

        # if the difference is significant, then draw borders
        if np.linalg.norm(palet1[i] - palet2[i]) > 0.02:
            arrow = mpatches.FancyArrowPatch((
               steps_x[0] + size_x,  steps_y[i]
            ), (
                steps_x[1] - size_x, steps_y[i]
            ), mutation_scale=20, color='black')
            ax.add_patch(arrow)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)

    #tensor_im = th.from_numpy(img).permute(-1, 0,1)
    #write_png(tensor_im, os.path.join(path, f'{prefix}_palet_th.png'))

    plt.savefig(os.path.join(path, f'{prefix}_palet_.png'), bbox_inches='tight', pad_inches=0.)
    plt.close()

def palette_to_img(palet1, path, prefix: str = ''):
    size = [800, 200, 3]
    img = np.ones((size), dtype=np.float32)

    midpoints = size[0] // (palet1.shape[0] + 1)
    steps_y = np.arange(start=midpoints, stop=size[0], step=midpoints)
    size_x = steps_y[0] // 2 - 20

    steps_x = size[1] // 2
    size_x = min(size_x, steps_x // 2 - 20)
    plot, ax = plt.subplots(1,1)

    for i in range(palet1.shape[0]):
        img[steps_y[i] - size_x - 2:steps_y[i] + size_x + 3,
            steps_x - size_x - 2:steps_x + size_x + 3] = np.zeros(3)
        img[steps_y[i] - size_x:steps_y[i] + size_x,
            steps_x - size_x:steps_x + size_x] = palet1[i]


    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)

    plt.savefig(os.path.join(path, f'{prefix}_palet.png'), bbox_inches='tight', pad_inches=0.)
    plt.close()

if __name__ == '__main__':
    p1 = th.rand((4,3), dtype=th.float32)
    p2 = p1.clone()
    p2[3] = th.rand((3), dtype=th.float32)
    palette_change_to_img(palet1=p1,
                          palet2=p2,
                          path='test_images')
    palette_to_img(palet1=p1, path='test_images')