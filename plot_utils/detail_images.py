from torchvision.io import read_image, write_png
from torchvision.transforms import Resize, CenterCrop
import os
import torch as th

def detail_img(img_str, path1, path2, size):
    img1 = read_image(os.path.join(path1, img_str))
    img2 = read_image(os.path.join(path2, img_str))

    diff = th.linalg.norm((img1.float() - img2.float()), dim=0)

    max_error_patch = []
    max_val = -1.
    step = 200

    for i in range(0, diff.shape[0], step // 2):
        for j in range(0, diff.shape[1], step // 2):
            error_val = diff[i:i+step, j:j+step].sum()

            if error_val > max_val:
                max_error_patch = [img1[:, i:i+step, j:j+step], img2[:, i:i+step, j:j+step]]
                max_val = error_val

    t = Resize(size=(step*2, step*2))
    img1_t = t(max_error_patch[0])
    img2_t = t(max_error_patch[1])

    # set the corner to red
    img1[:, img1.shape[1]-step*2-1:, img1.shape[1]-step*2-1:] = th.tensor((255, 0, 0), device=img1.device)[..., None, None]
    img2[:, img1.shape[1]-step*2-1:, img1.shape[1]-step*2-1:] = th.tensor((255, 0, 0), device=img1.device)[..., None, None]

    # set the corner to red
    img1[:, img1.shape[1] - step*2:, img1.shape[1] - step*2:] = img1_t
    img2[:, img1.shape[1] - step*2:, img1.shape[1] - step*2:] = img2_t

    img_combined = th.cat((img1_t, img2_t), dim=-1)
    img_combined[..., img_combined.shape[-1] // 2 - 1:  img_combined.shape[-1] // 2 + 1] = th.tensor((255, 0, 0), device=img1.device)[..., None, None]

    write_png(img1, os.path.join(path1, img_str[:img_str.find('.')] + '_detail.png'))
    write_png(img2, os.path.join(path2, img_str[:img_str.find('.')] + '_detail.png'))

    write_png(img_combined, os.path.join(path1, img_str[:img_str.find('.')] + '_combined.png'))
    write_png(img_combined, os.path.join(path2, img_str[:img_str.find('.')] + '_combined.png'))

if __name__ == '__main__':

    img_str = '005.png'
    path1 = os.path.join('ablation', 'palette_reg_with', "val_distill_nerf")
    path2 = os.path.join('ablation', 'palette_reg_without', "val_distill_nerf")
    size = 256

    detail_img(img_str,
               path1=path1,
               path2=path2,
               size=size)