import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, write_png
import torchvision.models as models
import torchvision.transforms as transforms
import os
import torchvision.transforms.functional as tf

import plot_utils.io as utility
from configargparse import Namespace

""" VGG-19 Architecture
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (17): ReLU(inplace=True)
  (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (24): ReLU(inplace=True)
  (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (26): ReLU(inplace=True)
  (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (31): ReLU(inplace=True)
  (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (33): ReLU(inplace=True)
  (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (35): ReLU(inplace=True)
  (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""

class StyleNetwork(nn.Module):
    def __init__(self, target_img: th.Tensor, params: Namespace, device='cuda', size=256, style_layers=[11]):
        super(StyleNetwork, self).__init__()
        self.device = device
        self.cnn_normalization_mean = th.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
        self.cnn_normalization_std = th.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)

        self.style_layers = style_layers
        max_layer = max(self.style_layers)
        self.vgg = models.vgg19(pretrained=True).features[:(max_layer + 1)].eval().to(self.device)

        for i, layer in enumerate(self.vgg.children()):
            if isinstance(layer, nn.ReLU):
                self.vgg[i] = nn.ReLU(inplace=False)

        self.size = size
        self.reshape = transforms.Resize(size=(self.size, self.size))
        self.random_crop = transforms.RandomCrop(size=(self.size, self.size), pad_if_needed=True)
        self.center_crop = transforms.CenterCrop(size=(self.size, self.size))
        self.use_center_crop = False
        self.preserve_color = params.preserve_color
        self.target_feature_color = None

        if target_img is None:
            assert params.style_image is not None
            self.image = self.load_image(os.path.join("style_images", params.style_image)).to(self.device)
        else:
            self.image = target_img.to(self.device)
        write_png((self.image.cpu() * 255).byte(),
                  os.path.join(params.ablation_dir, params.ablation_folder, 'style_image.png'))
        if self.image.shape[1] is not self.image.shape[2]:
            write_png((self.center_crop(self.image).cpu() * 255).byte(),
                      os.path.join(params.ablation_dir, params.ablation_folder, 'style_image_centercrop.png'))
        self.feature = self.image_to_feature(self.transform_image(self.image)).detach()
        self.gram_style = self.gram_matrix(self.feature)

        self.reshape_to_feat = None

    def match_color(self, target_img, eps=1e-5):
        # code adapted from
        # https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ColourControl.ipynb

        # this is actually the style image
        mu_t = self.image.mean(axis=(1, 2), keepdim=True)
        t = (self.image - mu_t).flatten(1, 2)
        Ct = t @ t.T / t.shape[1] + eps * th.eye(t.shape[0], device=t.device)

        if len(target_img.shape) == 2:
            target_img = target_img[..., None]

        # this is actually the target image
        mu_s = target_img.mean(axis=(1, 2), keepdim=True)
        s = (target_img - mu_s).flatten(1, 2)
        Cs = s @ s.T / s.shape[1] + eps * th.eye(s.shape[0], device=s.device)

        eva_t, eve_t = th.linalg.eigh(Ct)
        Qt = eve_t @ th.sqrt(th.diag(eva_t)) @ eve_t.T
        eva_s, eve_s = th.linalg.eigh(Cs)
        Qs = eve_s @ th.sqrt(th.diag(eva_s)) @ eve_s.T
        ts = Qs @ th.linalg.inv(Qt) @ t

        matched_img = ts.reshape(self.image.shape)
        matched_img += mu_s
        matched_img = th.clamp(matched_img, 0, 1)

        # set color_target_feature
        self.target_feature_color = self.gram_matrix(self.image_to_feature(matched_img)).detach()

        return matched_img

    def load_image(self, target_img: str):
        assert os.path.exists(target_img)
        return read_image(target_img) / 255.

    def transform_image(self, image: th.Tensor, crop=True):
        if crop:
            image = self.random_crop(image)
        else:
            image = self.reshape(image)
            # if self.use_center_crop else self.reshape(image)
        return (image - self.cnn_normalization_mean) / self.cnn_normalization_std

    def image_to_feature(self, x: th.tensor):
        outs = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.style_layers:
                outs.append(x)

        return th.stack(outs)

    def gram_matrix(self, input):
        a, b, c, d = input.size()

        outs = []
        for i in range(a):
            features = input[i].view(b, c * d)

            G = features @ features.T
            G = G.div(b * c * d)

            outs.append(G)
        return th.stack(outs)

    # experimental feature, not used
    def guided_gram_mse_loss(self, style_feature, target_img, guide_img):
        # first, random crop the same region from both target image and guide image (to ensure consistency)
        # using the torchvision functional API
        i, j, h, w = self.random_crop.get_params(target_img, output_size=(self.random_crop.size))
        cut_img = self.transform_image(tf.crop(target_img, i, j, h, w), crop=False)
        cut_guide = tf.crop(guide_img[None, ...], i, j, h, w)

        # reshape guide to size of the features
        if self.reshape_to_feat is None:
            self.reshape_to_feat = transforms.Resize(size=(style_feature.shape[-2:]))
        cut_guide = self.reshape_to_feat(cut_guide)

        # multiply both feature maps with the guide
        # obtain the gram matrix
        f, a, b, c = style_feature.size()
        G_style = th.einsum('zfab, zgab->fg', style_feature * cut_guide, style_feature * cut_guide).div(a * b * c)
        target_feat = self.image_to_feature(cut_img)
        G_img = th.einsum('zfab, zgab->fg', target_feat * cut_guide, target_feat * cut_guide).div(a * b * c)

        return F.mse_loss(G_style, G_img)

    # forward pass used for style transfer
    def forward(self, input_img, guide=None):
        # create a feature from the image
        x = self.image_to_feature(self.transform_image(input_img, crop=False))
        G = self.gram_matrix(x)
        if self.preserve_color:
            return F.mse_loss(G, self.target_feature_color)
        else:
            if guide is not None:
                return self.guided_gram_mse_loss(self.feature, input_img, guide)
            return F.mse_loss(G, self.gram_style)