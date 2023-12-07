from torchvision.io import read_image, write_png
from torchvision.transforms import Resize, CenterCrop
import os

def resize_img(img_str, path, size):
    img = read_image(img_str)

    t = Resize(size=(size, size))
    img_t = t(img)

    write_png(img_t, os.path.join(path,
                                  img_str[:img_str.find('.')] + '_resized.png'))

def center_crop_img(img_str, path, size):
    img = read_image(img_str)

    t = CenterCrop(size=(size, size))
    img_t = t(img)

    write_png(img_t, os.path.join(path,
                                  img_str[:img_str.find('.')] + '_centercrop.png'))

if __name__ == '__main__':

    img_str = 'circles.jpg'
    path = os.path.join('ablation', 'test_circles')
    size = 256

    resize_img(img_str,
               path=path,
               size=size)
    center_crop_img(img_str,
               path=path,
               size=size)