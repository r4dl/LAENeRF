import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, write_png
import torchvision.models as models
import torchvision.transforms as transforms

from encoding import get_encoder
from activation import trunc_exp
from ffmlp import FFMLP
import tinycudann as tcnn
import os
from editing.style_network import StyleNetwork

import plot_utils.io as utility
from configargparse import Namespace


class LAENeRF(nn.Module):
    def __init__(self,
                 params: Namespace,
                 encoding="hashgrid",
                 dir_encoding=None,
                 num_layers=3,
                 hidden_dim=64,
                 color_palette=None,
                 size=256,
                 style_img=None
                 ):
        super().__init__()
        # input point encoding
        self.opt = params
        self.bound = params.bound
        # TODO: with new gpu, use more capacity...
        self.encoder, self.in_dim = get_encoder(encoding,
                                                desired_resolution=2048 * self.bound, num_levels=16,
                                                log2_hashmap_size=19)

        th.seed()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_color_bases = self.opt.num_palette_bases

        self.active_palets = th.ones((self.num_color_bases), dtype=th.bool).cuda()
        if color_palette is not None:
            self.color_palette = color_palette
        else:
            self.color_palette = th.rand((self.num_color_bases, 3), dtype=th.float32).cuda()
        self.color_palette.requires_grad = True
        self.original_color_palette = None

        self.size = size

        self.dir_encoding = None
        self.in_dim_dir = 0
        if dir_encoding is not None:
            self.dir_encoding, self.in_dim_dir = get_encoder(dir_encoding, degree=3)

        if self.opt.style_weight > 0:
            self.style_transfer_net = StyleNetwork(target_img=style_img, params=params, device='cuda', size=self.size,
                                                   style_layers=self.opt.style_layers)

        # offset network
        self.offset_net = tcnn.Network(
            n_input_dims=self.in_dim + self.in_dim_dir,
            n_output_dims=(3),
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers - 1,
            },
        )

        # weight network
        self.weight_net = tcnn.Network(
                n_input_dims=self.in_dim,
                n_output_dims=(self.num_color_bases),
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
        )

    def color_transfer_style_img(self, target_img):
        self.style_transfer_net.match_color(target_img)

    def get_weights(self, x):
        x = self.encoder(x, bound=self.bound)
        w_hat = self.weight_net(x)[:, self.active_palets]
        return th.softmax(w_hat, -1)

    def get_offsets(self, x, d):
        x = self.encoder(x, bound=self.bound)

        offset_in = x
        if self.dir_encoding is not None:
            assert d is not None
            enc_d = self.dir_encoding(d)
            offset_in = th.cat([offset_in, enc_d], dim=-1)

        o_hat = self.offset_net(offset_in)
        return o_hat

    def forward(self, x, d=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], normalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        offset_in = x
        if self.dir_encoding is not None:
            assert d is not None
            enc_d = self.dir_encoding(d)
            offset_in = th.cat([offset_in, enc_d], dim=-1)

        w_hat = self.weight_net(x)[:, self.active_palets]
        o_hat = self.offset_net(offset_in)

        o_hat = th.tanh(o_hat)
        w_hat = th.softmax(w_hat, -1)

        # original formulation
        # pred_colors = (self.color_palette.T[:, None, :].half() * w_hat).permute(1, 2, 0) + o_hat
        # a different formulation: weighing the offsets with the color palette
        pred_colors = w_hat @ self.color_palette[self.active_palets].half() + o_hat
        return th.clamp(pred_colors, 0, 1)

    def forward_train(self, x, d=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], normalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        offset_in = x
        if self.dir_encoding is not None:
            assert d is not None
            enc_d = self.dir_encoding(d)
            offset_in = th.cat([offset_in, enc_d], dim=-1)

        w_hat = self.weight_net(x)[:, self.active_palets]
        o_hat = self.offset_net(offset_in)

        o_hat = th.tanh(o_hat)
        w_hat = th.softmax(w_hat, -1)

        # original formulation
        # pred_colors = (self.color_palette.T[:, None, :].half() * w_hat).permute(1, 2, 0) + o_hat
        # a different formulation: weighing the offsets with the color palette
        pred_colors = w_hat @ self.color_palette[self.active_palets].half() + o_hat
        return th.clamp(pred_colors, 0, 1), w_hat, o_hat

    def distill_color_palettes(self, edit_dataset, n: int = 10, thresh: float = 0.025):
        # sample n poses, extract weights, set to 0 where the weights are smaller than a threshold
        indices = th.randint(0, high=len(edit_dataset), size=(n,))#, generator=self.gen)
        weights = th.zeros((self.num_color_bases), dtype=th.float32).cuda()

        for idx in indices:
            i = idx.item()
            x_term = edit_dataset._data.x_term[i].cuda()
            pred_weights = self.get_weights(x_term)
            weights += pred_weights.mean(0)

        weights /= n
        self.active_palets = weights >= thresh

    def get_color_palette(self):
        return self.color_palette[self.active_palets]

    def set_color_palette(self, palet):
        if self.original_color_palette is None:
            self.original_color_palette = self.color_palette.detach().clone()

        with th.no_grad():
            self.color_palette[self.active_palets] = palet

    def style_loss(self, input_img, guide=None) -> th.tensor:
        assert self.style_transfer_net is not None
        return self.style_transfer_net.forward(input_img, guide=guide)

    def weights_loss(self, pred_bary_weights, params):
        # TODO: change the formulation, this causes NaN for a lot of values
        uniform_loss = th.sum(pred_bary_weights, dim=0).max()
        non_uniform_loss = (1 - pred_bary_weights.max(dim=-1).values).sum()
        return uniform_loss * params.weight_loss_uniform + non_uniform_loss * params.weight_loss_non_uniform
        # return uniform_loss - (pred_bary_weights.shape[0] / pred_bary_weights.shape[1]) + non_uniform_loss

    def palet_loss(self, params):
        # max distance between bases
        dists = (th.pow(self.color_palette[:, None, :] - self.color_palette, 2)).sum(-1)
        dist_loss = (1 - dists / dists.max()).mean()

        # minimize dist to out-of-gamut colors
        valid_loss = (th.floor(self.color_palette) * self.color_palette).sum()
        return valid_loss * params.palette_loss_valid + dist_loss * params.palette_loss_distinct

    def offset_loss(self, pred_offsets, params):
        return th.pow(pred_offsets, 2).sum() * params.offset_loss

    def tv_loss(self, output_image) -> th.tensor:
        w_var = th.sum((output_image[:, :-1, :] - output_image[:, 1:, :])**2)
        v_var = th.sum((output_image[..., :-1] - output_image[..., 1:])**2)
        return w_var + v_var

    def depth_discontinuity_loss(self, output_image, depth_v_var, depth_w_var) -> th.tensor:
        depth_v_var = (depth_v_var / depth_v_var.max())
        depth_w_var = (depth_w_var / depth_w_var.max())

        w_var = (th.pow(output_image[:, :-1, :] - output_image[:, 1:, :], 2) * depth_w_var[None, ...])
        v_var = (th.pow(output_image[..., :-1] - output_image[..., 1:], 2) * depth_v_var[None, ...])

        return -w_var.sum() - v_var.sum()

    def tv_loss_depth_weighted(self, output_image, depth_v_var, depth_w_var, weights_trans=None) -> th.tensor:
        if weights_trans is not None:
            depth_v_var = (1 - depth_v_var) * (1 - weights_trans[:, 1:])
            depth_w_var = (1 - depth_w_var) * (1 - weights_trans[1:, :])
        else:
            depth_v_var = (1 - depth_v_var)
            depth_w_var = (1 - depth_w_var)

        w_var = th.sum((th.pow(output_image[:, :-1, :] - output_image[:, 1:, :], 2) * depth_w_var[None, ...]))
        v_var = th.sum((th.pow(output_image[..., :-1] - output_image[..., 1:], 2) * depth_v_var[None, ...]))
        return w_var + v_var

    def smooth_transition_loss(self, ref_img, output_image, transition_weights):
        diff = th.pow(output_image - ref_img, 2).sum(-1)
        return (diff * transition_weights).sum()

    def intensity_loss(self, ref_img, output_image):
        return th.pow(th.linalg.norm(output_image, dim=-1, ord=2) - th.linalg.norm(ref_img, dim=-1, ord=2), 2).sum()

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.weight_net.parameters(), 'lr': lr},
            {'params': self.offset_net.parameters(), 'lr': lr},
            {'params': self.color_palette, 'lr': 2 * lr},
        ]

        return params
    def get_params_but_dont_learn_palette(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.weight_net.parameters(), 'lr': lr},
            {'params': self.offset_net.parameters(), 'lr': lr},
            {'params': self.color_palette, 'lr': 0},
        ]

        return params