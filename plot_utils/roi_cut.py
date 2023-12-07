from torchvision.io import read_image, write_png
from torchvision.transforms import Resize, CenterCrop
import os
import torch as th

def detail_img(path1, size):

    images = []
    names = []

    for i in os.listdir(path1):
        images.append(read_image(os.path.join(path1, i)))
        names.append(os.path.join(path1, i))

    img1 = images[0]
    img2 = images[1]

    diff = th.linalg.norm((img1.float() - img2.float()), dim=0)

    max_error_patch = []
    max_val = -1.
    step = 300

    i_, j_ = 0, 0

    for i in range(0, diff.shape[0], step // 6):
        for j in range(0, diff.shape[1], step // 6):
            error_val = diff[i:i+step, j:j+step].sum()

            if error_val > max_val:
                max_error_patch = [img1[:, i:i+step, j:j+step], img2[:, i:i+step, j:j+step]]
                i_, j_ = i, j
                max_val = error_val

    for i in range(len(images)):
        img = images[i]
        str_ = names[i]
        write_png(img[..., i_:i_+step, j_:j_+step], os.path.join(str_[:str_.find('.')] + '_detail.png'))

if __name__ == '__main__':

    path1 = os.path.join('test_images', 'diff_images')
    size = 256

    detail_img(path1=path1,
               size=size)