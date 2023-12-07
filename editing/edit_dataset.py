import os
import numpy as np
import torch as th

from nerf.utils import get_rays
from nerf.provider import NeRFDataset
from torch.utils.data import DataLoader
from torchvision.io import write_png
import open3d as o3d
import matplotlib.colors as mc
import cv2

class EditDataset:
    # depth_diff = .85 works well for fortress
    # depth_diff = .65 works well for flower
    # depth_diff = .50 works well for synthetic
    def __init__(self, opt, train_dataset: NeRFDataset, edit_grid, grow_grid, trainer, num_steps=512, max_dist=.12,
                 depth_diff=.65):
        super().__init__()

        self.opt = opt
        self.device = 'cuda'
        self.train_dataset = train_dataset
        self.egrid = edit_grid
        self.edit_grid = edit_grid.get_grid()
        self.nerf = trainer.model
        self.trainer = trainer
        self.nerf.eval()
        self.num_steps = num_steps
        self.max_dist = max_dist
        self.depth_diff = depth_diff

        self.w8s = []
        self.targets = []
        self.x_term = []
        self.dirs = []
        self.depths = []
        self.indices = []
        self.indices_interp = []
        self.dist_weights = []

        # for distillation
        self.weights_densitygrid = []
        self.weights_editgrid = []
        self.pred_imgs = []
        self.depth_factor = []

        # for the style losses and the TV losses
        self.cut_gt = []
        self.cut_weights = []
        self.cut_min_max_xy = []
        self.cut_tv_h = []
        self.cut_tv_v = []
        self.cut_smooth_trans = []

        self.occluded = []

        self.grow_grid = grow_grid.get_grid()
        # if this grid is None, set the smooth transition weight to 0
        if self.grow_grid is None:
            self.opt.smooth_trans_weight = 0

        if self.opt.load_edit_dataset is not None:
            self.load(self.opt.load_edit_dataset)
            return

        h, w = self.train_dataset._data.H, self.train_dataset._data.W
        intrinsics = self.train_dataset._data.intrinsics

        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "styleenc_train_dataset")
        if not os.path.exists(path):
            os.mkdir(path)

        for i in range(len(train_dataset)):
            pose_idx = i
            pose = self.train_dataset._data.poses[pose_idx].unsqueeze(0).to(self.device)
            rays = get_rays(pose, intrinsics, h, w, -1)
            data = {
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'H': h,
                'W': w,
                'images': self.train_dataset._data.images[pose_idx]
            }
            with th.cuda.amp.autocast(enabled=True):
                pred_img, pred_term, pred_w8s_density, pred_w8s_edit, depth_, min_near, depth_edit = trainer.distill_step(data,
                    self.edit_grid, perturb_depth=False)

            pred_w8s = pred_w8s_edit.clone()
            # difference must be larger than a threshold value \tau_{\text{weight}}
            # this eliminates the effect of floaters to a large degree
            # weights_edit_sum[torch.abs(weights_sum - weights_edit_sum) > 0.5] = 0
            pred_w8s[th.abs(pred_w8s_density - pred_w8s) > self.depth_diff] = 0
            # depth must be valid for the sample to be accepted
            # bigger than the minimal near bound
            # weights_edit_sum[depth.squeeze() < nears.min()] = 0
            pred_w8s[depth_.squeeze() < min_near] = 0
            # weights_edit_sum[weights_edit_sum > 0] = weights_sum[weights_edit_sum > 0]
            pred_w8s[pred_w8s > 0] = pred_w8s_density[pred_w8s > 0]

            mask = pred_w8s.nonzero(as_tuple=True)

            if pred_w8s.nonzero().nelement() == 0:
                # if the mask is empty, the region is completely occluded
                # we need to handle this case though :(
                # continue
                self.occluded.append(pose_idx)
                continue

            self.weights_densitygrid.append(pred_w8s_density.cpu())
            self.weights_editgrid.append(pred_w8s.cpu())
            self.pred_imgs.append(pred_img.cpu())

            write_png((pred_w8s * 255).reshape(self.train_dataset._data.images[pose_idx].shape[:2])[
                          None, ...].byte().detach().cpu(), os.path.join(path, f'weights_{i:03d}.png'))
            write_png((pred_w8s_edit * 255).reshape(self.train_dataset._data.images[pose_idx].shape[:2])[
                          None, ...].byte().detach().cpu(), os.path.join(path, f'weights_edit_{i:03d}.png'))

            if opt.smooth_trans_weight > 0:
                assert self.grow_grid is not None
                with th.cuda.amp.autocast(enabled=True):
                    pred_img_, pred_term_, _, pred_w8s_edit__, _, _, _ = trainer.distill_step(data, self.grow_grid,
                                                                                   perturb_depth=False, grow_grid=True)

                x_term_grow_grid = pred_term_[pred_w8s_edit__ > .99]
                pred_img_grow_grid = pred_img_[pred_w8s_edit__ > .99]
                if x_term_grow_grid.shape[0]:
                    stp = int(1e3)
                    min_dists = None
                    for i in range(0, pred_term.shape[0], stp):
                        dists = th.linalg.norm(pred_term[mask][i:i+stp, None, :] - x_term_grow_grid[None, ...], dim=-1)
                        if min_dists is not None:
                            min_dists = th.cat([min_dists, th.min(dists, dim=-1).values])
                        else:
                            min_dists = th.min(dists, dim=-1).values
                    min_dists = th.clamp_max(min_dists, max_dist)

                    dist_factor = 1 - (min_dists / min_dists.max())
                else:
                    dist_factor = th.zeros_like(pred_w8s[mask])
                mask_dist = dist_factor.nonzero(as_tuple=True)
                self.indices_interp.append(mask_dist[0].cpu())
                self.dist_weights.append(dist_factor[mask_dist].cpu())

            target = self.train_dataset._data.images[pose_idx]
            if target.shape[-1] == 4:
                target = target[..., :3] * target[..., -1][..., None]
            target = target.reshape(-1, 3)

            #write_png(
            #    ((pred_w8s[..., None].cpu() * target.cpu()) * 255).reshape(self.train_dataset._data.images[pose_idx].shape).permute(
            #        -1, 0, 1).byte().detach().cpu(), os.path.join(path, f'target_{i:03d}.png'))

            #test = th.zeros((data['H'], data['W'])).cuda()
            #test = data['images'][..., -1].cuda()
            #test.flatten(0,1)[mask] = 0.
            #test.flatten(0,1)[mask[0][mask_dist]] = dist_factor[mask_dist]
            #write_png((test * 255).byte().cpu()[None, ...], 'test.png')

            #c1 = (th.tensor(mc.to_rgb(mc.CSS4_COLORS['cadetblur'])) * 255).byte().cuda()
            #c2 = (th.tensor(mc.to_rgb(mc.CSS4_COLORS['coral'])) * 255).byte().cuda()

            #weight_img = (test[..., None] * c1 + (1- test[..., None]) * c2).cpu().byte()
            # because we use cv, we need to transpose the channels...
            #temp_weight_w_alpha = th.cat([weight_img[..., [2,1,0]],  (data['images'][..., -1] * 255).byte()[..., None]], dim=-1)
            # write with cv2 (we use image 2 from the chair scene)
            #cv2.imwrite('hmm.png', temp_weight_w_alpha.numpy())

            d_mask = depth_[mask]
            d_ = th.zeros((self.train_dataset._data.images[pose_idx].shape[:2])).cuda()
            d_.flatten(0,1)[mask] = d_mask
            write_png(((d_ - d_.min()) / (d_.max() - d_.min()) * 255).byte().cpu()[None, ...],
                      os.path.join(path, f'depth_{i:03d}.png') )

            pred_w8s = pred_w8s[mask]
            target = target.cuda()
            target = target[mask]

            #pointcloud = o3d.geometry.PointCloud()
            #pointcloud.points = o3d.utility.Vector3dVector(pred_term[mask].cpu().numpy())
            #pointcloud.colors = o3d.utility.Vector3dVector(pred_img[mask].cpu().numpy())
            #o3d.io.write_point_cloud('pointcloud_test1.ply', pointcloud)

            self.w8s.append(pred_w8s.cpu())
            self.targets.append(target.cpu())
            self.x_term.append(pred_term[mask].cpu())
            self.dirs.append(data['rays_d'].squeeze()[mask].cpu())
            self.depths.append(d_mask.cpu())
            self.indices.append(mask[0].cpu())

            # precompute everything necessary
            m = th.zeros((h, w), dtype=th.float32).cuda()
            m.flatten(0, 1)[mask] = pred_w8s
            x, y = m.nonzero(as_tuple=True)
            x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
            self.cut_min_max_xy.append(th.tensor((x_min, x_max, y_min, y_max)))

            # precompute ground truth image cutout
            ground_truth = th.zeros((h, w, 3), dtype=th.float32).cuda()
            ground_truth.flatten(0, 1)[mask] = target.float()
            ground_truth = ground_truth[x_min:x_max, y_min:y_max]
            self.cut_gt.append(ground_truth.cpu())

            # precompute weights
            weights = th.zeros((h, w), dtype=th.float32).cuda()
            weights.flatten(0, 1)[mask] = pred_w8s
            weights = weights[x_min:x_max, y_min:y_max]
            weights[weights < 0.98] = 0
            w_h = (weights[:-1, :] * weights[1:, :])
            w_h[1:] *= weights[:-2, :] * weights[2:, :]
            w_v = (weights[:, :-1] * weights[:, 1:])
            w_v[:, 1:] *= weights[:, :-2] * weights[:, 2:]

            # precompute RGB diff
            rgb_h_var = th.abs(ground_truth[:-1, :] - ground_truth[1:, :]).sum(-1).cuda()
            rgb_v_var = th.abs(ground_truth[:, :-1] - ground_truth[:, 1:]).sum(-1).cuda()

            depth = th.zeros((h, w), dtype=th.float32).cuda()
            depth.flatten(0, 1)[mask] = d_mask
            depth = depth[x_min:x_max, y_min:y_max]

            depth_h_var = th.abs(depth[:-1, :] - depth[1:, :]) * w_h * rgb_h_var
            depth_v_var = th.abs(depth[:, :-1] - depth[:, 1:]) * w_v * rgb_v_var
            self.cut_tv_h.append(depth_h_var.cpu())
            self.cut_tv_v.append(depth_v_var.cpu())

            if self.opt.smooth_trans_weight > 0:
                weights_trans = th.zeros((data['H'], data['W']), dtype=th.float32).cuda()
                weights_trans.flatten(0, 1)[mask] = dist_factor.float()
                weights_trans = weights_trans[x_min:x_max, y_min:y_max]
                self.cut_smooth_trans.append(weights_trans.cpu())

            self.depth_factor.append((d_mask.max() - d_mask.min()) / self.num_steps)
        self.save()

    def save(self):
        save_dict = {
            'device': self.device,
            'num_steps': self.num_steps,
            'max_dist': self.max_dist,
            'depth_diff': self.depth_diff,
            'w8s': self.w8s,
            'targets': self.targets,
            'x_term': self.x_term,
            'dirs': self.dirs,
            'depths': self.depths,
            'indices': self.indices,
            'indices_interp': self.indices_interp,
            'dist_weights': self.dist_weights,
            'weights_densitygrid': self.weights_densitygrid,
            'weights_editgrid': self.weights_editgrid,
            'pred_imgs': self.pred_imgs,
            'depth_factor': self.depth_factor,
            'cut_gt': self.cut_gt,
            'cut_weights': self.cut_weights,
            'cut_min_max_xy': self.cut_min_max_xy,
            'cut_tvh': self.cut_tv_h,
            'cut_tvv': self.cut_tv_v,
            'cut_smooth_trans': self.cut_smooth_trans,
            'occluded': self.occluded
        }
        th.save(save_dict, os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, 'edataset.pth'))

    def load(self, where):
        save_dict = th.load(os.path.join(where, 'edataset.pth'))
        self.device = save_dict['device']
        self.num_steps = save_dict['num_steps']
        self.depth_diff = save_dict['depth_diff']
        self.w8s = save_dict['w8s']
        self.targets = save_dict['targets']
        self.x_term = save_dict['x_term']
        self.dirs = save_dict['dirs']
        self.depths = save_dict['depths']
        self.indices = save_dict['indices']
        self.indices_interp = save_dict['indices_interp']
        self.dist_weights = save_dict['dist_weights']
        self.weights_densitygrid = save_dict['weights_densitygrid']
        self.weights_editgrid = save_dict['weights_editgrid']
        self.pred_imgs = save_dict['pred_imgs']
        self.depth_factor = save_dict['depth_factor']
        self.cut_gt = save_dict['cut_gt']
        self.cut_weights = save_dict['cut_weights']
        self.cut_min_max_xy = save_dict['cut_min_max_xy']
        self.cut_tv_h = save_dict['cut_tvh']
        self.cut_tv_v = save_dict['cut_tvv']
        self.cut_smooth_trans = save_dict['cut_smooth_trans']
        self.occluded = save_dict['occluded']

    def collate(self, index):
        i = index[0]

        # randomly add some noise in the ray direction
        x_term = self.x_term[i].cuda()
        dirs = self.dirs[i].cuda()
        d_ = (th.rand(x_term.shape[0], device=x_term.device) - 0.5) * self.depth_factor[i]
        x_term += d_[..., None] * dirs

        results = {
            'w8s': (self.w8s[i]).cuda(),
            'x_term': x_term,
            'target': (self.targets[i]).cuda(),
            'depth': self.depths[i].cuda(),
            'd': dirs,
            'indices': tuple([(self.indices[i]).cuda()]),
            'depth_h_var': self.cut_tv_h[i].cuda(),
            'depth_v_var': self.cut_tv_v[i].cuda(),
            'minmax': self.cut_min_max_xy[i],
            'cut_gt': self.cut_gt[i].cuda(),
            'cut_smooth': self.cut_smooth_trans[i].cuda() if self.opt.smooth_trans_weight > 0 else None
        }

        return results

    def dataloader(self):
        size = len(self.targets)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=True,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = True
        return loader