import os

import torch
import torch as th
import json
from icecream import ic
import cv2
import numpy as np

from nerf.utils import get_rays
from nerf.provider import NeRFDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.io import write_png, read_image

import plot_utils.io as io

class SingleViewEditDataset:
    def __init__(self, opt, train_dataset: NeRFDataset, trainer, config_str: str, sem_encoder, num_steps=512,
                 min_dist=1e-2, max_dist=10e-2):
        super().__init__()

        self.opt = opt
        self.device = 'cuda'
        self.train_dataset = train_dataset
        self.nerf = trainer.model
        self.trainer = trainer
        self.semantic_encoder = sem_encoder
        self.nerf.eval()

        assert config_str is not None
        self.config = None
        with open(os.path.join(config_str, 'data_config.json'), 'r') as f:
            self.config = json.load(f)

        self.num_steps = num_steps
        self.min_dist = min_dist
        self.max_dist = max_dist

        # to get all images from the config directory
        ref_img_str = [os.path.join(config_str, i) for i in os.listdir(config_str) if any(x in i for x in ['.png', '.jpg', 'jpeg'])]
        # just load the first one ...
        self.ref_img = read_image(ref_img_str[0]).float()
        self.ref_img /= 255.

        # for .png images
        if self.ref_img.shape[0] == 4:
            self.ref_img = self.ref_img[:3] * self.ref_img[-1][None, ...]

        self.train_dataset = train_dataset

        self.w8s = []

        self.targets = []
        self.targets_gt = []
        self.x_term = []
        self.dirs = []
        self.origins = []
        self.depths = []
        self.indices = []

        self.indices_ray_reg = []
        self.w8s_ray_reg = []

        self.depth_factor = []
        self.use_style_loss = []
        self.content_loss_mult =[]
        self.style_img = None

        self.style_feat = None
        self.content_feat = None
        self.sup_feats = []
        self.col_patches = []
        self.style_guides = []
        self.target_weigths = []

        # solely for distillation
        self.weights_editgrid = []
        self.pred_imgs = []

        # for the style losses and the TV losses
        self.cut_gt = []
        self.cut_weights = []
        self.cut_min_max_xy = []
        self.cut_tv_h = []
        self.cut_tv_v = []

        self.h, self.w = self.train_dataset._data.H, self.train_dataset._data.W
        self.intrinsics = self.train_dataset._data.intrinsics

        self.path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "styleenc_train_dataset")
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.weight_imgs = []
        self.ref_imgs = []
        self.style_imgs = []
        self.weight_depths = []
        self.depth_imgs = []

        self.extract_ref_views()

    def extract_ref_views(self):
        # First, obtain x_term and rgb from the target (stylized) image
        pose_idx = self.config['tmpl_idx_train']
        pose = self.train_dataset._data.poses[pose_idx].unsqueeze(0).to(self.device)
        rays = get_rays(pose, self.intrinsics, self.h, self.w, -1, perturb_ray_dirs=False)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': self.h,
            'W': self.w,
            'images': self.train_dataset._data.images[pose_idx]
        }
        with th.cuda.amp.autocast(enabled=True):
            pred_img, pred_term, pred_w8s_density, _, depth_, min_near = self.trainer.distill_step(data,
                                                                                                   self.trainer.model.density_bitfield,
                                                                                                   perturb_depth=False,
                                                                                                   grow_grid=True)

        mask = self.train_dataset._data.images[pose_idx, ..., -1].cuda().flatten()
        mask[mask > 0] = 1
        mask = mask.nonzero(as_tuple=True)

        test = th.zeros((1, self.h, self.w)).cuda()
        test.flatten()[mask] = pred_w8s_density[mask]
        write_png((test * 255).byte().cpu(), os.path.join(self.path, f't_ref.png'))

        ref_x_term = pred_term[mask]
        ref_rgb = self.ref_img.permute(1, 2, 0).flatten(0, 1).cuda()[mask]
        ref_dirs = rays['rays_d'][0][mask]

        # cut rgb
        m = th.zeros((self.h, self.w), dtype=th.float32).cuda()
        m.flatten(0, 1)[mask] = pred_w8s_density[mask]
        x, y = m.nonzero(as_tuple=True)
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()

        ref_img = th.zeros((self.h, self.w, 3), dtype=th.float32).cuda()
        ref_img.flatten(0,1)[mask] = ref_rgb.cuda()
        write_png((ref_img * 255).permute(-1, 0, 1).byte().cpu(), os.path.join(self.path, f'img_ref.png'))
        ref_img = ref_img.permute(-1, 0, 1)
        self.style_img = ref_img[:, x_min:x_max, y_min:y_max].cuda()

        self.style_feat = self.semantic_encoder.encode_feats(ref_img[:, x_min:x_max, y_min:y_max],
                                                             layers=[11, 13, 15],
                                                             size=(self.opt.feature_size, self.opt.feature_size)).detach()

        # content feature
        # resized to (256, 256), taking layers [11, 13, 15]
        content_feature_im = self.train_dataset._data.images[pose_idx, ..., :3].cuda().permute(-1, 0, 1)
        self.content_feat = self.semantic_encoder.encode_feats(content_feature_im[:, x_min:x_max, y_min:y_max],
                                                               layers=[11, 13, 15],
                                                               size=(self.opt.feature_size, self.opt.feature_size)).detach()

        # color feature
        # not resized, taking layers [25, 27, 29]
        color_feature_im = self.semantic_encoder.encode_feats(content_feature_im,
                                                              layers=[25, 27, 29], size=None).detach()
        # size: 3x50x50 as color feature is not resized
        patch_mean_color = self.semantic_encoder.get_mean_patch_color(img=ref_img, size=(color_feature_im.shape[-2],
                                                                                         color_feature_im.shape[-1]))

        # more views results in better details
        # we perturb ray directions for these extra rays to obtain a more diverse set of samples
        for _ in range(2):
            rays = get_rays(pose, self.intrinsics, self.h, self.w, -1, perturb_ray_dirs=True)
            data = {
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'H': self.h,
                'W': self.w,
                'images': self.train_dataset._data.images[pose_idx]
            }
            with th.cuda.amp.autocast(enabled=True):
                pred_img, pred_term, pred_w8s_density, _, depth_, min_near = self.trainer.distill_step(data,
                                                                                                       self.trainer.model.density_bitfield,
                                                                                                       perturb_depth=False,
                                                                                                       grow_grid=True)

            mask = self.train_dataset._data.images[pose_idx, ..., -1].cuda().flatten()
            mask[mask > 0] = 1
            mask = mask.nonzero(as_tuple=True)

            ref_x_term = th.cat([ref_x_term, pred_term[mask]])
            ref_rgb = th.cat([ref_rgb, self.ref_img.permute(1, 2, 0).flatten(0, 1).cuda()[mask]])
            ref_dirs = th.cat([ref_dirs, rays['rays_d'][0][mask]])

        # then, iterate over all training pixels
        for i in range(len(self.train_dataset)):
            pose_idx = i
            pose = self.train_dataset._data.poses[pose_idx].unsqueeze(0).to(self.device)
            rays = get_rays(pose, self.intrinsics, self.h, self.w, -1, perturb_ray_dirs=False)
            data = {
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'H': self.h,
                'W': self.w,
                'images': self.train_dataset._data.images[pose_idx]
            }
            with th.cuda.amp.autocast(enabled=True):
                pred_img, pred_term, pred_w8s_density, _, depth_, min_near = self.trainer.distill_step(data,
                                                                               self.trainer.model.density_bitfield,
                                                                               perturb_depth=False, grow_grid=True)

            mask = self.train_dataset._data.images[pose_idx, ..., -1].cuda().flatten()
            mask[mask > 0] = 1
            mask = mask.nonzero(as_tuple=True)

            self.weights_editgrid.append(pred_w8s_density.cpu())
            self.pred_imgs.append(pred_img.cpu())

            pred_xterm = pred_term[mask]

            target_ = self.train_dataset._data.images[pose_idx].cuda()
            if target_.shape[-1] == 4:
                target_ = target_[..., :3] * target_[..., -1][..., None]
            target_ = target_.reshape(-1, 3)[mask]

            min_dist, mask_dist, target, target_weights = self.get_ref_supervision(pred_xterm=pred_xterm,
                                                                                   ref_x_term=ref_x_term,
                                                                                   ref_rgb=ref_rgb,
                                                                                   min_dist_ref=self.min_dist,
                                                                                   ref_dirs=ref_dirs,
                                                                                   dirs=rays['rays_d'][0][mask])

            self.target_weigths.append(th.clamp_min(target_weights, 0).cpu())

            # obtain a mask for stylization based on the distance to the closest termination point
            # GOAL: a smooth transition from 1e-2 to 10e-2
            mask_style = th.clamp(min_dist, self.min_dist, self.max_dist)
            mask_style = (mask_style - self.min_dist) / (self.max_dist - self.min_dist)
            mask_style = th.clamp_min(mask_style, self.opt.min_tv_factor)

            d_mask = depth_[mask]
            if i < 20:
                test = th.zeros_like(self.train_dataset._data.images[pose_idx]).cuda()[..., :3]
                test.flatten(0, 1)[mask[0][mask_dist]] = target.cuda() * target_weights[..., None]
                write_png((test * 255).byte().cpu().permute(-1,0,1), os.path.join(self.path, f'ref_{i:03d}.png'))

            pred_w8s = pred_w8s_density[mask]

            self.targets.append(target.cpu())
            self.targets_gt.append(target_.cpu())
            self.w8s.append(pred_w8s.cpu())
            self.x_term.append(pred_xterm.cpu())
            self.dirs.append(data['rays_d'].squeeze().cpu())
            self.origins.append(data['rays_o'].squeeze().cpu())
            self.depths.append(d_mask.cpu())
            self.indices.append(mask[0].cpu())
            self.indices_ray_reg.append(mask_dist[0].cpu())

            self.depth_factor.append((d_mask.max() - d_mask.min()) / self.num_steps)

            # precompute everything necessary
            m = th.zeros((self.h, self.w), dtype=th.float32).cuda()
            m.flatten(0, 1)[mask] = pred_w8s
            x, y = m.nonzero(as_tuple=True)
            x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
            self.cut_min_max_xy.append(th.tensor((x_min, x_max, y_min, y_max)))

            # precompute ground truth image cutout
            ground_truth = th.zeros((self.h, self.w, 3), dtype=th.float32).cuda()
            ground_truth.flatten(0, 1)[mask] = target_
            ground_truth_resized = ground_truth[x_min:x_max, y_min:y_max]
            self.cut_gt.append(ground_truth_resized.cpu())

            # precompute the supervision feature
            sup_feat = self.semantic_encoder.encode_feats(ground_truth_resized.permute(-1, 0, 1),
                                                          layers=[11, 13, 15],
                                                          size=(self.opt.feature_size, self.opt.feature_size))
            # replace with the closest features according to style
            sup_feat_nn = self.semantic_encoder.nn_feat_replace(sup_feat, self.content_feat, self.style_feat)

            col_feat = self.semantic_encoder.encode_feats(self.train_dataset._data.images[pose_idx, ..., :3].cuda().permute(-1, 0, 1),
                                                          layers=[25, 27, 29], size=None).detach()
            col_feat_nn = self.semantic_encoder.nn_feat_replace_color(col_feat, color_feature_im, patch_mean_color)

            self.sup_feats.append(sup_feat_nn.detach().cpu())
            self.col_patches.append(col_feat_nn.detach().cpu())
            if i < 20:
                r = torchvision.transforms.Resize(size=(512, 512))
                f_r = r(col_feat_nn)
                write_png((f_r * 255).byte().cpu(), os.path.join(self.path, f'patch_{i:03d}.png'))

            # also, reshape the mask for style guidance
            style_guidance = th.zeros((self.h, self.w)).cuda()
            style_guidance.flatten()[mask] = mask_style
            style_guidance = style_guidance[x_min:x_max, y_min:y_max]
            self.style_guides.append(style_guidance.detach().cpu())
            if i < 20:
                write_png((style_guidance * 255).byte().cpu()[None, ...],
                          os.path.join(self.path, f'style_mask_{i:03d}.png'))

            # precompute weights
            weights = th.zeros((self.h, self.w), dtype=th.float32).cuda()
            weights.flatten(0, 1)[mask] = pred_w8s
            weights = weights[x_min:x_max, y_min:y_max]
            weights[weights < 0.98] = 0
            w_h = (weights[:-1, :] * weights[1:, :])
            w_h[1:] *= weights[:-2, :] * weights[2:, :]
            w_v = (weights[:, :-1] * weights[:, 1:])
            w_v[:, 1:] *= weights[:, :-2] * weights[:, 2:]

            # precompute RGB diff
            rgb_h_var = th.abs(ground_truth_resized[:-1, :] - ground_truth_resized[1:, :]).sum(-1).cuda()
            rgb_v_var = th.abs(ground_truth_resized[:, :-1] - ground_truth_resized[:, 1:]).sum(-1).cuda()

            depth = th.zeros((self.h, self.w), dtype=th.float32).cuda()
            depth.flatten(0, 1)[mask] = d_mask
            depth = depth[x_min:x_max, y_min:y_max]

            depth_h_var = th.abs(depth[:-1, :] - depth[1:, :]) * w_h * rgb_h_var
            depth_v_var = th.abs(depth[:, :-1] - depth[:, 1:]) * w_v * rgb_v_var
            self.cut_tv_h.append(depth_h_var.cpu())
            self.cut_tv_v.append(depth_v_var.cpu())

    def get_ref_supervision(self, pred_xterm, ref_x_term, ref_rgb, min_dist_ref, ref_dirs=None, dirs=None):
        min_dist = th.zeros((pred_xterm.shape[0]), dtype=th.float32).cuda()
        argmin_dist = th.zeros((pred_xterm.shape[0]), dtype=th.int32).cuda()
        step = int(1e3)

        for z in range(0, pred_xterm.shape[0], step):
            min_dist[z:z + step], argmin_dist[z:z + step] = th.linalg.norm(pred_xterm[z:z + step, None, :] - ref_x_term,
                                                                           axis=-1).min(-1)

        # distance to the nearest must be smaller than 1e-2
        mask_dist = (min_dist < min_dist_ref).nonzero(as_tuple=True)
        # if true, use this as supervision
        # pred_xterm = pred_xterm[mask_dist]
        target = ref_rgb[argmin_dist[mask_dist].long()].clone()

        target_weights = min_dist[mask_dist]
        # map weights to interval [0, 1]
        target_weights = (target_weights - target_weights.min()) / (target_weights.max() - target_weights.min())
        # now, the rays which are very far have the most weights, so invert
        target_weights = th.abs(target_weights - 1.)

        if dirs is not None and ref_dirs is not None:
            # only the selected target directions
            target_dirs = ref_dirs[argmin_dist[mask_dist].long()].clone()
            # only registered ray dirs
            dirs = dirs[mask_dist]
            dist = th.nn.functional.cosine_similarity(target_dirs, dirs, dim=1)
            # this is now in range [-1, +1], with -1 diff. dir and 1 lin. dependent
            # down-weigh the target_weights based on this distance
            dist_factor = (th.clamp(dist, min=-1, max=-0.5) + 1) / 0.5
            target_weights *= dist_factor

        return min_dist, mask_dist, target, target_weights

    def collate(self, index):
        i = index[0]

        inds = tuple([(self.indices[i]).cuda()])
        # randomly add some noise in the ray direction
        x_term = self.x_term[i].cuda()
        dirs = self.dirs[i].cuda()[inds]
        d_ = (th.rand(x_term.shape[0], device=x_term.device) - 0.5) * self.depth_factor[i]
        x_term += d_[..., None] * dirs

        results = {
            'w8s': (self.w8s[i]).cuda(),
            'target_feat': self.sup_feats[i].cuda(),
            'color_patch': self.col_patches[i].cuda(),
            'style_feat': self.style_feat.cuda(),
            'style_guide': self.style_guides[i].cuda(),
            'content_loss_mult': 1.,
            'x_term': x_term,
            'target': (self.targets[i]).cuda(),
            'target_weights': self.target_weigths[i].cuda(),
            'depth': self.depths[i].cuda(),
            'd': dirs,
            'indices': tuple([(self.indices_ray_reg[i]).cuda()]),
            'indices_': tuple([(self.indices[i][self.indices_ray_reg[i]]).cuda()]),
            'indices__': inds,
            'use_style_loss': False,
            'depth_h_var': self.cut_tv_h[i].cuda(),
            'depth_v_var': self.cut_tv_v[i].cuda(),
            'minmax': self.cut_min_max_xy[i],
            'cut_gt': self.cut_gt[i].cuda(),
            'cut_smooth': None
        }

        return results

    def collate_gt(self, index):
        i = index[0]

        inds = tuple([(self.indices[i]).cuda()])
        # randomly add some noise in the ray direction
        x_term = self.x_term[i].cuda()
        dirs = self.dirs[i].cuda()[inds]
        d_ = (th.rand(x_term.shape[0], device=x_term.device) - 0.5) * self.depth_factor[i]
        x_term += d_[..., None] * dirs

        results = {
            'w8s': (self.w8s[i]).cuda(),
            'content_loss_mult': 1.,
            'x_term': x_term,
            'target': (self.targets_gt[i]).cuda(),
            'depth': self.depths[i].cuda(),
            'd': dirs,
            'indices': inds,
            'use_style_loss': False,
            'depth_h_var': self.cut_tv_h[i].cuda(),
            'depth_v_var': self.cut_tv_v[i].cuda(),
            'minmax': self.cut_min_max_xy[i],
            'cut_gt': self.cut_gt[i].cuda(),
            'cut_smooth': None
        }

        return results

    def collate_nerf_train(self, index):
        i = index[0]

        inds = torch.randint(0, (self.h * self.w), size=[self.train_dataset._data.opt.num_rays]).cuda()  # may duplicate

        results = {
            'target': self.ref_imgs[i].flatten(0,1).cuda()[inds][None, ...],
            'target_weights': self.weight_imgs[i].cuda().flatten()[inds][None, ..., None],
            'depth': self.depth_imgs[i].cuda().flatten(0,1)[inds][None, ...],
            'depth_weights': self.weight_depths[i].cuda().flatten(0,1)[inds][None, ...],
            'rays_o': self.origins[i].cuda()[inds][None, ...],
            'rays_d': self.dirs[i].cuda()[inds][None, ...],
            'style_img': self.style_imgs[i].cuda().flatten(0,1)[inds][None, ...]
        }

        return results

    def dataloader(self):
        size = len(self.targets)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=True,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader

    def dataloader_gt(self):
        size = len(self.targets)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_gt, shuffle=True,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader

    def dataloader_nerf(self, style_encoder):

        self.path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "nerf_retrain_dataset")
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # we create a new dataset based on the style encoder
        for i in range(len(self.train_dataset)):
            train_image_cuda = self.train_dataset._data.images[i, ...].cuda()

            pred_x_term = (self.x_term[i]).cuda()
            indices = (self.indices[i]).cuda()
            depth = self.depths[i].cuda()

            dirs = self.dirs[i].cuda()[indices]
            indices_ray_reg = tuple([(self.indices[i][self.indices_ray_reg[i]]).cuda()])
            target = (self.targets[i]).cuda()
            target_weights = self.target_weigths[i].cuda()

            target_weight = th.zeros((self.h, self.w)).cuda()
            target_weight.flatten()[indices_ray_reg] = target_weights
            # also add weight to where alpha = 0
            target_weight += (1 - train_image_cuda[..., -1])
            self.weight_imgs.append(target_weight.cpu().detach())
            write_png((target_weight[None, ...] * 255).byte().cpu(), os.path.join(self.path, f'target_weight_{i}.png'))

            # TODO: handle palette interpolation
            #if self.opt.smooth_trans_weight > 0:
            #    indices_interp = tuple([self.indices_interp[i].cuda()])
            #    dist_weights = self.dist_weights[i].cuda()

            train_image_gpu = th.zeros_like(train_image_cuda)
            # transfer alpha from train image
            train_image_gpu[..., -1] = train_image_cuda[..., -1]
            # transfer ray registration
            train_image_gpu.flatten(0,1)[..., :3][indices_ray_reg] = target
            self.ref_imgs.append(train_image_gpu.cpu().detach())
            write_png((train_image_gpu[..., :3] * 255).permute(-1, 0, 1).byte().cpu(),
                      os.path.join(self.path, f'train_img_{i}.png'))

            # TODO: handle palette interpolation
            pred_colors = style_encoder(
                pred_x_term,
                d=dirs
            )

            style_image_gpu = th.zeros_like(train_image_cuda)
            # transfer alpha
            style_image_gpu.flatten(0, 1)[tuple(indices), -1] = (
                train_image_cuda.flatten(0, 1))[tuple(indices), -1]
            # transfer predicted colors
            style_image_gpu.flatten(0, 1)[tuple(indices), :3] = pred_colors.float()
            self.style_imgs.append(style_image_gpu.cpu().detach())
            write_png((style_image_gpu[..., :3].permute(-1, 0, 1) * 255).byte().cpu(),
                      os.path.join(self.path, f'style_img_{i}.png'))

            depth_ref = th.zeros((self.h, self.w)).cuda()
            depth_ref.flatten(0,1)[indices] = depth
            self.depth_imgs.append(depth_ref.cpu().detach())
            d_write = depth_ref.clone()
            d_write = (d_write - d_write.min()) - (d_write.max() - d_write.min())

            write_png((d_write[None, ...] * 255).byte().cpu(), os.path.join(self.path, f'depth_ref_{i}.png'))

            depth_weight = th.zeros((self.h, self.w)).cuda()
            depth_weight.flatten(0,1)[indices] = 1.
            self.weight_depths.append(depth_weight.cpu())
            write_png((depth_weight[None, ...] * 255).byte().cpu(), os.path.join(self.path, f'depth_w8_{i}.png'))


        ic('finished with the distilled nerf dataset')

        size = len(self.targets)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_nerf_train, shuffle=True,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader