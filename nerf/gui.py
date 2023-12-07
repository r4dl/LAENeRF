import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
import torchvision
from scipy.spatial.transform import Rotation as R
from editing.style_encoder import StyleEncoder
from editing.edit_dataset import EditDataset
from editing.single_view_edit_dataset import SingleViewEditDataset
from editing.editgrid import EditGrid
from editing.semantic_encoder import SemanticEncoder

import json
from icecream import ic
import imageio
import cv2

from .utils import *
from plot_utils import palette_utils as pu

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
    

class NeRFGUI:
    def  __init__(self, opt, trainer, train_loader=None, val_loader=None, video_loader=None, test_loader=None,
                 debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step
        self.distill_step = 0
        self.distill = False
        self.distill_npr = False
        self.train_styleenc = False
        self.train_styleenc_npr = False
        self.trainfast = False

        self.npr_string = None
        self.extracted_npr = False

        self.grid = EditGrid()
        self.negative_grid = EditGrid()
        self.growing_grid = EditGrid()
        self.current_grid = self.grid
        self.style_encoder: StyleEncoder = None
        self.semantic_encoder: SemanticEncoder = None
        self.edit_dataset = None
        self.edit_dataset_ = None
        self.pose = None
        self.timings = []

        self.starter, self.ender = None, None

        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.video_loader = video_loader
        self.test_loader = test_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16
        self.show_grid = 'density' # should be one of [density, edit, grow]
        self.select_grid_pos = False
        self.selected_points = []
        self.project_points = False
        self.growing_steps = 5
        self.growing_iterations = 5000
        self.grow_reg = False
        self.render_vid = False

        self.palet = torch.zeros((self.opt.num_palette_bases, 3), dtype=torch.float32)
        self.original_palet = torch.zeros((self.opt.num_palette_bases, 3), dtype=torch.float32)

        self.palet_weights = torch.ones((self.opt.num_palette_bases), dtype=torch.float32)
        self.palet_biases = torch.zeros((self.opt.num_palette_bases), dtype=torch.float32)

        self.highlight_palette_id = 0
        self.show_baryweights: bool = False
        self.show_offsets: bool = False
        self.use_offsets: bool = True
        self.train_view = 0
        self.show_styleenc = self.train_styleenc or self.train_styleenc_npr

        self.style_size = self.opt.crop_size
        self.style_img_set = False
        self.style_img = np.zeros((self.style_size, self.style_size, 3), dtype=np.float32)
        self.style_img_original = None
        self.style_img_scaled = None
        self.style_img_offsets = np.zeros(2, dtype=np.uint32)

        dpg.create_context()
        self.register_dpg()
        self.loss_distill = []

        self.grow_grid_str = None
        self.edit_grid_str = None

        if os.path.exists(os.path.join(self.opt.workspace, 'edit_grid.pth')):
            self.grid.load_grid_as_torch(os.path.join(self.opt.workspace, 'edit_grid.pth'))
        if os.path.exists(os.path.join(self.opt.workspace, 'grow_grid.pth')):
            self.growing_grid.load_grid_as_torch(os.path.join(self.opt.workspace, 'grow_grid.pth'))
        else:
            if self.opt.run_all:
                self.opt.smooth_trans_weight = 0.
            print(self.opt.smooth_trans_weight)
        self.eval = False
        self.eval_type = 'None'

        self.dataloader_npr = None
        self.timer_train = None

        if self.opt.run_all:
            if self.opt.style_enc_path is not None:
                # distill directly
                self.distill = True
                self.trainfast = True
                self.style_encoder = torch.load(self.opt.style_enc_path)
                self.show_styleenc = True
                self.palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()
                self.original_palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()

                if self.opt.palette_path is not None:
                    self.palet = torch.load(self.opt.palette_path)
                    self.sync_with_styleenc()

                self.distill = True
            else:
                if self.opt.ref_npr_config is None:
                    self.train_styleenc = True
                else:
                    self.train_styleenc_npr = True
                    self.npr_string = os.path.join('single_view_stylization', self.opt.ref_npr_config)
    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        if self.distill_npr:
            if self.dataloader_npr is None:
                self.dataloader_npr = self.edit_dataset_.dataloader_nerf(style_encoder=self.style_encoder)
            outputs = self.trainer.train_gui_npr(self.train_loader, distill=self.distill, step=self.train_steps,
                                                 dataloader=self.dataloader_npr)
        else:
            outputs = self.trainer.train_gui(self.train_loader, distill=self.distill, step=self.train_steps,
                                             depth_sup=self.opt.style_weight > 0, edit_dataset=self.edit_dataset)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        if not self.trainfast:
            self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def init_style_pred(self,
                        style_img,
                        use_palette=False):
        # init the edit dataset
        if self.edit_dataset is None:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if self.train_styleenc_npr:
                self.semantic_encoder = SemanticEncoder()
                self.edit_dataset_ = SingleViewEditDataset(self.opt, self.train_loader, self.trainer,
                                                           self.npr_string, self.semantic_encoder,
                                                           min_dist=self.opt.reg_max_dist,
                                                           max_dist=self.opt.tv_min_dist)
                self.edit_dataset = self.edit_dataset_.dataloader_gt()
            else:
                self.edit_dataset = EditDataset(self.opt, self.train_loader, self.grid, self.growing_grid,
                                                self.trainer, depth_diff=self.opt.depth_diff).dataloader()

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            print(f"Extracting the Training dataset took {t / 1000.:.2f} seconds!")

            self.timings.append(t / 1000.)

        # init the style predictor
        if self.style_encoder is None:
            self.style_encoder = StyleEncoder(
                params=self.opt,
                color_palette=None,
                dir_encoding="sphere_harmonics" if not self.train_styleenc_npr else None,
                #rgb_encoding="frequency",
                style_img=style_img if not self.train_styleenc_npr else self.edit_dataset._data.style_img,
                size=self.opt.crop_size
            ).cuda()

        palette_losses = {
            "weights": {
                "weight_loss_uniform": self.opt.weight_loss_uniform,
                "weight_loss_non_uniform": self.opt.weight_loss_non_uniform,
            },
            "offsets": {
                "offset_loss": self.opt.offset_loss
            },
            "palette": {
                "palette_loss_valid": self.opt.palette_loss_valid,
                "palette_loss_distinct": self.opt.palette_loss_distinct,
                "num_palette_bases": self.opt.num_palette_bases,
            },
            #"npr": {
            #    "npr": self.train_styleenc_npr,
            #    "config": self.opt.ref_npr_config,
            #    "reg_max_dist": self.opt.reg_max_dist,
            #    "tv_min_dist": self.opt.tv_min_dist,
            #    "cos_loss_factor": self.opt.cos_loss_factor,
            #    "mse_loss": self.opt.mse_loss,
            #    "color_patch_loss": self.opt.color_patch_loss,
            #    "min_tv_factor": self.opt.min_tv_factor,
            #    "style_weight_d": self.opt.style_weight_d,
            #    "depth_weight_d": self.opt.depth_weight_d,
            #}
        }

        style_losses = {
            "style_image": self.opt.style_image,
            "style_weight": self.opt.style_weight,
            "style_layers": self.opt.style_layers,
            "TV": {
                "tv_weight": self.opt.tv_weight,
                "depth_disc_weight": self.opt.depth_disc_weight,
                "tv_depth_guide": self.opt.tv_depth_guide,
            },
            "misc": {
                "intensity_weight": self.opt.intensity_weight,
                "smooth_trans_weight": self.opt.smooth_trans_weight,
                "style_image": self.opt.style_image,
                "train_steps_style": self.opt.train_steps_style,
                "train_steps_distill": self.opt.train_steps_distill,
                "preserve_color": self.opt.preserve_color,
                "warmup_iterations": self.opt.warmup_iterations,
                "feature_size": self.opt.feature_size,
            },
            "path_options": {
                "ablation_dir": self.opt.ablation_dir,
                "ablation_folder": self.opt.ablation_folder,
            }
        }

        hparams = {
            "palette_losses": palette_losses,
            "style_losses": style_losses,
        }
        with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "hparams.json"), "w") as outfile:
            json.dump(hparams, outfile, indent=2)
        with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "opt.json"), "w") as outfile:
            json.dump(vars(self.opt), outfile, indent=2)

        # SAVE the edit grid and the growing grid
        torch.save(self.grid.get_grid(),
                   f=os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, 'edit_grid.pth'))
        if self.growing_grid.get_grid() is not None:
            torch.save(self.growing_grid.get_grid(),
                       f=os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, 'grow_grid.pth'))

    def train_style_step(self,
                         train_steps=16):
        outputs = self.trainer.train_styleenc_step(self.train_loader,
                                                   edit_dataset=self.edit_dataset,
                                                   style_encoder=self.style_encoder,
                                                   params=self.opt,
                                                   step=train_steps,
                                                   global_step=self.distill_step)
        self.loss_distill.append(outputs["loss"])
        self.distill_step += train_steps

    def train_style_step_npr(self,
                         train_steps=16):
        t_s = min(len(self.edit_dataset), train_steps)

        if self.distill_step > self.opt.warmup_iterations and not self.extracted_npr:
            ic('now train dataset with ref-ray matching')
            self.edit_dataset = self.edit_dataset_.dataloader()
            self.extracted_npr = True

        outputs = self.trainer.train_styleenc_step_npr(self.train_loader,
                                                       edit_dataset=self.edit_dataset,
                                                       style_encoder=self.style_encoder,
                                                       params=self.opt,
                                                       step=t_s,
                                                       global_step=self.distill_step,
                                                       semantic_encoder=self.semantic_encoder)
        self.loss_distill.append(outputs["loss"])
        self.distill_step += t_s

    def distill_dataset(self,
                        train_dataset,
                        style_encoder: StyleEncoder,
                        edit_dataset,
                        save_train_dataset=False,
                        blend_thresh=0.5):
        # overwrite each image of the training dataset with our expected outputs from the style encoder
        # but likely needs more supervision in terms of expected ray termination

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        print(f"Started distilling the edits!")

        H, W = train_dataset._data.H, train_dataset._data.W
        style_encoder.eval()

        palet_og = self.style_encoder.original_color_palette
        if palet_og is not None:
            palet_og = palet_og[self.style_encoder.active_palets]
        else:
            palet_og = self.style_encoder.get_color_palette()
        palet_mod = self.style_encoder.get_color_palette()

        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder)
        if save_train_dataset:
            path = os.path.join(path, "train_dataset_mod")
            if not os.path.exists(path):
                os.mkdir(path)
            path_weights = os.path.join(path, 'weights')
            if not os.path.exists(path_weights):
                os.mkdir(path_weights)

        # save the palettes
        pu.palette_to_img(palet_og.cpu().detach(), path=path, prefix='original')
        pu.palette_to_img(palet_mod.cpu().detach(), path=path, prefix='modified')
        pu.palette_change_to_img(palet_og.cpu().detach(), palet_mod.cpu().detach(), path=path, prefix='mod')

        # losses lists
        tv_loss = []
        sp_loss = []

        # initialize error map
        if self.opt.use_error_maps:
            train_dataset._data.error_map = torch.ones([train_dataset._data.images.shape[0], 128 * 128], dtype=torch.float).cuda()

        num_occ = 0
        for idx in range(len(train_dataset)):
            if idx in edit_dataset._data.occluded:
                num_occ = num_occ + 1
                train_dataset._data.depths.append(torch.zeros_like(w8s_edit).flatten().cpu())
                continue
            i = idx - num_occ
            pred_x_term = (edit_dataset._data.x_term[i]).cuda()
            indices = (edit_dataset._data.indices[i]).cuda()
            w8s_edit = (edit_dataset._data.weights_editgrid[i]).cuda()[..., None]
            pred_img = (edit_dataset._data.pred_imgs[i]).cuda()
            depth = (edit_dataset._data.depths[i]).cuda()
            dirs = (edit_dataset._data.dirs[i]).cuda()
            if self.opt.smooth_trans_weight > 0:
                indices_interp = tuple([edit_dataset._data.indices_interp[i].cuda()])
                dist_weights = edit_dataset._data.dist_weights[i].cuda()

            if self.opt.use_error_maps:
                # test: reshape edit weights to error map...
                weight_img = torch.zeros((H, W)).cuda()
                weight_img.flatten(0,1)[...] = w8s_edit[..., 0]
                reshape = torchvision.transforms.Resize((128, 128))
                weight_img_reshaped = torch.clamp(reshape(weight_img[None, ...]) + 15e-2, 0, 1)
                train_dataset._data.error_map[idx] = weight_img_reshaped.flatten()

                write_png(((weight_img_reshaped).cpu() * 255).byte(),
                          os.path.join(path, f'error_map_{idx:03d}.png'))

            train_image_gpu = train_dataset._data.images[idx, ..., :3].cuda()
            style_image_gpu = torch.zeros_like(train_image_gpu).cuda()

            _, weights_og, offsets = style_encoder.forward_train(
                pred_x_term,
                d=dirs
            )

            p_biases = self.palet_biases.cuda()[:self.style_encoder.active_palets.sum().item()][None]
            p_weights = self.palet_weights.cuda()[:self.style_encoder.active_palets.sum().item()][None]

            # include user-guided weights and biases
            weights = torch.clamp_min(p_biases + p_weights * weights_og, 0)
            weights /= weights.sum(-1)[..., None].half()

            pred_colors = torch.clamp(offsets.half() + weights.half() @ palet_mod.half(), 0, 1)

            if (self.opt.smooth_trans_weight > 0) and (not torch.allclose(palet_mod, palet_og) or not torch.all(self.palet_weights == 1) or not torch.all(self.palet_biases == 0)):
                # for recoloring of palettes -> interpolate in palette space
                # interpolate the palette for these indices
                palet_interp = dist_weights[..., None, None] * palet_og[None, ...] + (1 - dist_weights[..., None, None]) * palet_mod[None, ...]
                weight_interp = weights_og[indices_interp] * dist_weights[..., None] + weights[indices_interp] * (1 - dist_weights[..., None])
                interp_reg = torch.clamp(torch.einsum('bi,bik->bk', weight_interp.half(), palet_interp.half()) + offsets[indices_interp], 0, 1)
                pred_colors[indices_interp] = interp_reg

            if self.opt.preload:
                style_image_gpu.flatten(0, 1)[tuple(indices), ...] = pred_colors
            else:
                style_image_gpu.flatten(0, 1)[tuple(indices), ...] = pred_colors.float()
            if self.opt.no_bg:
                style_image_gpu = w8s_edit.reshape(H, W, 1) * style_image_gpu
            else:
                style_image_gpu = (1 - (w8s_edit)).reshape(H, W, 1) * pred_img.reshape(H,W,-1) + w8s_edit.reshape(H,W,1) * style_image_gpu

            mask = (w8s_edit <= blend_thresh).reshape(H, W, -1)

            # composite with the ground truth image where not intersected with the edit grid
            train_image_gpu = torch.clamp(~mask * style_image_gpu + mask * train_image_gpu, min=0, max=1)

            train_dataset._data.images[idx, ..., :3] = train_image_gpu.detach().cpu()
            if save_train_dataset:
                if train_dataset._data.images[idx].shape[-1] == 4:
                    bg = torch.ones_like(train_dataset._data.images[idx][..., :3]).cuda() * self.bg_color.cuda()
                    alpha = train_dataset._data.images[idx][..., -1][..., None].cuda()
                    final_img = bg * (1 - alpha) + train_dataset._data.images[idx][..., :3].cuda() * alpha
                else:
                    final_img = train_image_gpu
                write_png(((final_img).cpu() * 255).byte().permute(-1, 0, 1),
                                os.path.join(path, f'train_{i:03d}.png'))
                # edit weights
                test = torch.clone(w8s_edit).reshape(H, W, -1)

                # sparsity loss
                sp_loss.append(((weights.sum(-1) / torch.pow(weights,2).sum(-1)) - 1).mean())

                # tv loss
                weight_imgs = torch.zeros((W * H, weights.shape[-1])).cuda()
                weight_imgs[indices] = weights.float()
                # reshape weight img
                weight_imgs = weight_imgs.reshape(H, W, -1)
                w8s_edit_r = w8s_edit.reshape(H, W, -1)
                tv_1 = torch.pow((weight_imgs[1:, ...] - weight_imgs[:-1, ...]) * w8s_edit_r[1:] * w8s_edit_r[:-1], 2).sum() / indices.shape[0]
                tv_2 = torch.pow((weight_imgs[:, 1:] - weight_imgs[:, :-1]) * w8s_edit_r[:, 1:] * w8s_edit_r[:, :-1], 2).sum() / indices.shape[0]
                tv_loss.append((tv_1 + tv_2))

                write_png(((test).cpu() * 255).byte().permute(-1, 0, 1), os.path.join(path, f'w8s_{i:03d}.png'))

                # save the modified color palette bases
                if idx % 10 == 0:
                    for w_i in range(self.palet.shape[0]):
                        out_normal = (torch.ones((W, H, 3), dtype=torch.float32) * self.bg_color).cuda()
                        out_normal.flatten(0, 1)[indices] = palet_mod[w_i, [2, 1, 0]].cuda()
                        out_n_alpha = (torch.zeros((W, H, 1), dtype=torch.float32)).cuda()
                        out_n_alpha.flatten(0, 1)[indices] = weights[:, w_i][..., None].float()
                        out_normal = torch.cat([out_normal, out_n_alpha], dim=-1)
                        cv2.imwrite(os.path.join(path_weights, f'{i:03d}_w{w_i:02d}.png'),
                                    (out_normal * 255).byte().cpu().numpy())

            # need to also save depth
            d_ = torch.zeros_like(w8s_edit).flatten()
            d_[indices] = depth
            train_dataset._data.depths.append(d_.cpu())

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        losses = {
            "sparsity_loss": (sum(sp_loss) / len(sp_loss)).item(),
            "tv_loss": (sum(tv_loss) / len(tv_loss)).item()
        }
        with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "palette_eval.json"), "w") as outfile:
            json.dump(losses, outfile, indent=2)

        self.timings.append(t / 1000.)
        if len(self.timings) == 3:
            self.timings.append(sum(self.timings))

            timings_ = {
                "edit_dataset": f'{self.timings[0]:.2f} s',
                "train_style_enc": f'{self.timings[1]:.2f} s',
                "distill_dataset": f'{self.timings[2]:.2f} s',
                "sum": f'{self.timings[-1]:.2f} s',
            }

            with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "timings.json"), "w") as outfile:
                json.dump(timings_, outfile, indent=2)

        print(f"Distilling the edits into the train dataset took {t / 1000.:.2f} seconds!")

        if train_dataset._data.error_map is not None:
            self.trainer.error_map = train_dataset._data.error_map

    def grow_region(self, thresh=None):
        self.selected_points = []
        dens_grid = self.trainer.model.density_grid

        if (self.current_grid is not None):
            # first: bitwise and with density grid
            self.current_grid.bw_and(self.trainer.model.density_bitfield)

            # then: perform region growing
            dens_thresh = self.trainer.model.density_thresh if thresh is None else thresh
            self.current_grid.grow_region_queue(dens_grid, dens_thresh, grow_iterations=self.growing_iterations)

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)


    def project_points_(self):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.project_points(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.selected_points,
                                              self.downscale)

        term = outputs['term']
        self.current_grid.new_from_points(term, trainer=self.trainer, bound=self.opt.bound)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        print(f"projection of points took {t / 1000.:2f} seconds")

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()



            pose = self.pose.cpu().numpy() if self.pose is not None else self.cam.pose
            outputs = self.trainer.test_gui(pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp,
                                            self.downscale,
                                            self.current_grid.get_grid() if self.show_grid == 'edit' else self.growing_grid.get_grid() if self.show_grid == 'grow' else None)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

    def test_step_styleenc(self):
        if self.need_update or self.spp < self.opt.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            pose = self.pose.numpy() if self.pose is not None else self.cam.pose

            outputs = self.trainer.test_gui_styleenc(pose, self.cam.intrinsics, self.W, self.H,
                                                     self.current_grid.get_grid() if self.npr_string is None else None,
                                                     self.style_encoder, self.bg_color, self.spp, self.downscale,
                                                     -1 if not (self.show_baryweights or self.show_offsets) else self.highlight_palette_id,
                                                     show_weights=self.show_baryweights, use_offsets=self.use_offsets,
                                                     p_weights=self.palet_weights.cuda()[:self.style_encoder.active_palets.sum().item()],
                                                     p_bias=self.palet_biases.cuda()[:self.style_encoder.active_palets.sum().item()])

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1 / 4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                # self.draw_gl_()
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

    def eval_style_predictor(self, loader, folder_name):
        h = loader._data.H
        w = loader._data.W
        intrinsics = loader._data.intrinsics

        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, folder_name)
        if not os.path.exists(path):
            os.mkdir(path)
        pu.palette_to_img(self.palet.cpu().detach(), path, prefix='original')
        path_normal = os.path.join(path, "output")
        if not os.path.exists(path_normal):
            os.mkdir(path_normal)
        # without the offsets
        path_offsets = os.path.join(path, "offsets")
        if not os.path.exists(path_offsets):
            os.mkdir(path_offsets)
        # show the weights every few epochs
        path_weights = os.path.join(path, "weights")
        if not os.path.exists(path_weights):
            os.mkdir(path_weights)

        # show the selection for each image
        path_selection = os.path.join(path, "selection")
        if not os.path.exists(path_selection):
            os.mkdir(path_selection)

        mod_palet = torch.clone(self.style_encoder.get_color_palette())
        index = torch.randint(low=0, high=self.palet.shape[0], size=(2,)).unique()
        mod_palet[index] = torch.rand((index.shape[0], 3)).cuda()

        pu.palette_to_img(mod_palet.cpu().detach(), path, prefix='modified')
        pu.palette_change_to_img(self.palet.cpu().detach(), mod_palet.cpu().detach(), path)

        # quick check if this is NeRFSynthetic or nah...
        is_nerf_synthetic: bool = self.train_loader._data.images[0].shape[-1] == 4

        for i in range(len(loader)):
            pose_idx = i
            pose = loader._data.poses[pose_idx].cpu().numpy()

            if h is None:
                img = loader._data.images[pose_idx]
                h = img.shape[0]
                w = img.shape[1]

            # just normally
            outputs = self.trainer.val_gui_styleenc(pose, intrinsics, w, h, bg_color=self.bg_color,
                                                     edit_grid=self.current_grid.get_grid() if self.npr_string is None else None,
                                                     style_enc=self.style_encoder)
            alpha = outputs['alpha'][outputs['indices']][..., None]

            cpred = (outputs['weights'] @ self.style_encoder.get_color_palette().half()) + outputs['offsets']
            cpred = torch.clamp(cpred, 0, 1)
            out_normal = (torch.ones((h, w, 3), dtype=torch.float32) * self.bg_color).cuda()
            out_normal.flatten(0, 1)[outputs['indices']] = cpred * alpha + out_normal.flatten(0, 1)[outputs['indices']] * (1 - alpha)
            write_png((out_normal * 255).byte().cpu().permute(-1, 0, 1), os.path.join(path_normal, f'{i:03d}.png'))

            # render offsets
            cpred = torch.clamp(outputs['offsets'] / 2 + 0.5, 0, 1)
            out_normal = (torch.ones((h, w, 3), dtype=torch.float32) * self.bg_color).cuda()
            out_normal.flatten(0, 1)[outputs['indices']] = cpred * alpha + out_normal.flatten(0, 1)[outputs['indices']] * (1 - alpha)
            write_png((out_normal * 255).byte().cpu().permute(-1, 0, 1), os.path.join(path_offsets, f'{i:03d}.png'))

            cpred = torch.clamp((outputs['offsets'] * 10) / 2 + 0.5, 0, 1)
            out_normal = (torch.ones((h, w, 3), dtype=torch.float32) * self.bg_color).cuda()
            out_normal.flatten(0, 1)[outputs['indices']] = cpred * alpha + out_normal.flatten(0, 1)[outputs['indices']] * (1 - alpha)
            write_png((out_normal * 255).byte().cpu().permute(-1, 0, 1), os.path.join(path_offsets, f'{i:03d}_.png'))

            binary = torch.zeros((h, w), dtype=torch.bool).cuda()
            binary.flatten()[outputs['indices']] = True
            # to numpy
            binary_np = binary.cpu().numpy()
            # find contours
            contours, hierarchy = cv2.findContours(binary_np.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            drawing = np.zeros((binary_np.shape[0], binary_np.shape[1], 3), np.uint8)
            # draw contours and hull points
            for z in range(len(contours)):
                color_contours = (0, 0, 255)  # green - color for contours
                # draw ith contour
                cv2.drawContours(drawing, contours, z, color_contours, 3, 8, hierarchy)
            br_params = cv2.boundingRect(contours[np.argmax([c.shape[0] for c in (contours)])])
            cv2.imwrite(os.path.join(path_selection, f'{i:03d}.png'),
                        np.concatenate([drawing, drawing[..., -1][..., None]], axis=-1))

            test_image_np = (loader._data.images[i] * 255).byte().cpu().numpy()
            if is_nerf_synthetic:
                test_image_np = np.ones((1, 1, 3)) * 255 * self.bg_color.cpu().numpy() * (1 - test_image_np[..., -1][..., None] / 255.) + (test_image_np[..., -1][..., None] / 255) * test_image_np[..., :3]
                test_image_np = test_image_np.astype(np.uint8)
            test_image_np = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR)
            blurred_test_image = cv2.cvtColor(test_image_np, cv2.COLOR_RGB2GRAY)
            blurred_test_image = cv2.GaussianBlur(blurred_test_image, (27, 27), 0)
            if not is_nerf_synthetic:
                blurred_test_image = (blurred_test_image.astype(float) * 0.45).astype(np.uint8)

            # write mask
            cv2.imwrite(os.path.join(path_selection, f'{i:03d}_mask.png'), (outputs['alpha'].reshape(h, w).cpu().numpy() * 255).astype(np.uint8))

            mask = cv2.GaussianBlur(outputs['alpha'].reshape(h, w).cpu().numpy(), (9, 9), 0)[..., None]
            final = test_image_np * mask + (1 - mask) * blurred_test_image[..., None]
            cv2.imwrite(os.path.join(path_selection, f'{i:03d}_bg.png'), final)

            crop_size = max(br_params[-2], br_params[-1])
            max_minus = min(min(10, br_params[1]), br_params[0])
            max_plus = min(final.shape[1] - (br_params[0]+crop_size),
                           min(final.shape[0] - (br_params[1]+crop_size), 10))
            cv2.imwrite(os.path.join(path_selection, f'{i:03d}_bg_crop.png'),
                        final[br_params[1]-max_minus:br_params[1]+crop_size+max_plus,
                              br_params[0]-max_minus:br_params[0]+crop_size+max_plus])
            # draw a bounding rectangle for composition
            drawing = np.zeros((binary_np.shape[0], binary_np.shape[1], 3), np.uint8)
            color_contours = (0, 0, 255)  # green - color for contours
            # draw ith contour
            cv2.rectangle(drawing, (br_params[0]-max_minus, br_params[1]-max_minus),
                              (br_params[0]+crop_size+max_plus, br_params[1]+crop_size+max_plus),
                              color_contours, 3)
            cv2.imwrite(os.path.join(path_selection, f'{i:03d}_rect.png'),
                        np.concatenate([drawing, drawing[..., -1][..., None]], axis=-1))
            with open(os.path.join(path_selection, f'{i:03d}_rect.json'), 'w') as outfile:
                json.dump({
                    'start': (br_params[0]-max_minus, br_params[1]-max_minus),
                    'size': crop_size+max_plus,
                    'end': (br_params[0]+crop_size+max_plus, br_params[1]+crop_size+max_plus)
                }, outfile, indent=2)

            # show the weights
            if i % 10 == 0:
                for w_i in range(self.palet.shape[0]):
                    a_ = outputs['weights'][:, w_i][..., None]
                    out_normal = (torch.ones((h, w, 3), dtype=torch.float32) * self.bg_color).cuda()
                    out_normal.flatten(0, 1)[outputs['indices']] = self.palet[w_i, [2, 1, 0]].cuda()
                    out_n_alpha = (torch.zeros((h, w, 1), dtype=torch.float32)).cuda()
                    out_n_alpha.flatten(0, 1)[outputs['indices']] = a_.float()
                    out_normal = torch.cat([out_normal, out_n_alpha], dim=-1)
                    cv2.imwrite(os.path.join(path_weights, f'{i:03d}_w{w_i:02d}.png'),
                                (out_normal * 255).byte().cpu().numpy())

        print("Finished rendering the validation dataset! (StyleEnc)")

    def eval_nerf(self, loader, prefix: str):
        if loader is None:
            return

        h = loader._data.H
        w = loader._data.W
        intrinsics = loader._data.intrinsics

        self.vid = False
        if prefix in 'video':
            self.frames = []
            self.vid = True

        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, f"{prefix}_distill_nerf")
        if not os.path.exists(path):
            os.mkdir(path)
        # write the palette change if any occurred
        if not torch.allclose(self.palet, self.original_palet):
            pu.palette_change_to_img(self.original_palet, self.palet, path, 'change_in')

        errors = []
        for i in range(len(loader)):
            pose_idx = i
            pose = loader._data.poses[pose_idx].cpu().numpy()

            if h is None:
                img = loader._data.images[pose_idx]
                h = img.shape[0]
                w = img.shape[1]

            outputs = self.trainer.test_gui(pose, intrinsics, w, h, bg_color=self.bg_color)

            if 'train' in prefix:
                truths = loader._data.images[pose_idx].cpu()
                psnr = -10 * torch.log10(torch.mean((torch.from_numpy(outputs['image']).float() - truths.cpu().float()) ** 2))
                errors.append(psnr)

            if self.vid:
                self.frames.append((outputs["image"].clip(0,1)*255).astype(np.uint8))
            write_png(torch.from_numpy(outputs["image"] * 255).byte().permute(-1, 0, 1),
                      os.path.join(path, f'{i:03d}.png'))
        print(f"Finished rendering the {prefix} dataset! (NeRF)")
        if 'train' in prefix:
            print(f"PSNR: {(sum(errors) / len(errors)).item()}")
            results_ = {
                'psnr':  (sum(errors) / len(errors)).item(),
                'psnrs': [e.item() for e in errors]
            }
            with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "results_psnr_train.json"),
                      "w") as outfile:
                json.dump(results_, outfile, indent=2)

        if self.vid:
            seq = np.stack(self.frames, axis=0)
            self.mwrite(os.path.join(path, 'vid.mp4'), seq)

    def eval_masked(self, loader1, prefix: str, loader2=None, loader3=None):
        if loader1 is None:
            return

        masks_exists = False

        for i in loader1._data.masks:
            if i is not None:
                masks_exists = True
        if loader2 is not None:
            for i in loader2._data.masks:
                if i is not None:
                    masks_exists = True
        if not masks_exists:
            ic('no masks found :(')
            return

        h = loader1._data.H
        w = loader1._data.W
        intrinsics = loader1._data.intrinsics

        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, f"{prefix}_distill_nerf")
        if not os.path.exists(path):
            os.mkdir(path)

        errors = []

        loader = loader1
        for i in range(len(loader)):
            pose_idx = i
            pose = loader._data.poses[pose_idx].cpu().numpy()

            if h is None:
                img = loader._data.images[pose_idx]
                h = img.shape[0]
                w = img.shape[1]
            mask = loader._data.masks[pose_idx]
            if mask is None:
                continue

            outputs = self.trainer.test_gui(pose, intrinsics, w, h, bg_color=self.bg_color)
            m = torch.from_numpy((1 - np.clip(mask[..., -1], 0, 1))[..., None])

            mse = (((torch.from_numpy(outputs['image']) - loader._data.images[pose_idx].cpu()) * m.cpu())**2).mean()
            errors.append(mse.item())

        loader = loader3
        for i in range(len(loader)):
            pose_idx = i
            pose = loader._data.poses[pose_idx].cpu().numpy()

            if h is None:
                img = loader._data.images[pose_idx]
                h = img.shape[0]
                w = img.shape[1]
            mask = loader._data.masks[pose_idx]
            if mask is None:
                continue

            outputs = self.trainer.test_gui(pose, intrinsics, w, h, bg_color=self.bg_color)
            m = torch.from_numpy((1 - np.clip(mask[..., -1], 0, 1))[..., None])

            mse = (((torch.from_numpy(outputs['image']) - loader._data.images[pose_idx].cpu()) * m.cpu()) ** 2).mean()
            errors.append(mse.item())

        loader = loader2
        for i in range(len(loader)):
            pose_idx = i
            pose = loader._data.poses[pose_idx].cpu().numpy()

            if h is None:
                img = loader._data.images[pose_idx]
                h = img.shape[0]
                w = img.shape[1]
            mask = loader._data.masks[pose_idx]
            if mask is None:
                continue

            outputs = self.trainer.test_gui(pose, intrinsics, w, h, bg_color=self.bg_color)
            m = torch.from_numpy((1 - np.clip(mask[..., -1], 0, 1))[..., None])

            mse = (((torch.from_numpy(outputs['image']) - loader._data.images[pose_idx].cpu()) * m.cpu())**2).mean()
            errors.append(mse.item())

        print(f"Finished rendering the {prefix} dataset! (NeRF)")
        ic(f'mean_error = {sum(errors) / len(errors):.5f}')

        results_ = {
            "errors": errors,
            "mean": sum(errors) / len(errors),
        }

        with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "results_mask.json"),
                  "w") as outfile:
            json.dump(results_, outfile, indent=2)

    def mwrite(self, filename, frames):
        frames = frames[:, :frames.shape[1] // 2 * 2, :frames.shape[2] // 2 * 2]
        imageio.mimwrite(filename, frames, fps=25, quality=8, macro_block_size=1)

    def screenshot(self):
        n = datetime.now()
        str_ = f'{n.year}{n.month:02d}{n.day:02d}_{n.hour:02d}{n.minute:02d}{n.second:02d}'
        write_png(torch.from_numpy(self.render_buffer * 255).byte().permute(-1, 0, 1),
                  f'screenshot_{str_}.png')
        print(f'{str_} saved')

        if self.palet is not None and self.original_palet is not None:
            pu.palette_change_to_img(
                palet1=self.original_palet.cpu(),
                palet2=self.palet.cpu(),
                path='.',
                prefix=str_
            )


    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_raw_texture(self.style_size, self.style_size, self.style_img, format=dpg.mvFormat_Float_rgb, tag="_style_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=600, height=800):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")
            with dpg.group(horizontal=True):
                dpg.add_text("", tag="_log_train_log")

            def callback_snap_to_view(sender, app_data):
                if app_data < len(self.train_loader):
                    self.train_view = app_data

            with dpg.group(horizontal=True):
                dpg.add_text("Snap to Train View")
                dpg.add_input_int(label="", default_value=0, max_value=len(self.train_loader), width=125,
                                  callback=callback_snap_to_view)

                def callback_snap(sender, app_data):
                    self.pose = self.train_loader._data.poses[self.train_view]
                    self.need_update = True

                def callback_reset_snap(sender, app_data):
                    self.pose = None
                    self.need_update = True

                dpg.add_button(label="Snap", tag="_button_snap", callback=callback_snap)
                dpg.bind_item_theme("_button_snap", theme_button)

                dpg.add_button(label="Reset", tag="_button_reset_snap", callback=callback_reset_snap)
                dpg.bind_item_theme("_button_reset_snap", theme_button)

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=False):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    """with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")
                    """

            
            # edit info
            with dpg.collapsing_header(label="Editing", default_open=False):
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_text("Current Grid: ")
                    def callback_select_grid(sender, app_data):
                        if int(app_data) == 0:
                            self.current_grid = self.grid
                        elif int(app_data) == 1:
                            self.current_grid = self.negative_grid
                        else:
                            self.current_grid = self.growing_grid
                        self.need_update = True

                    dpg.add_slider_int(label="", min_value=0, max_value=2, format="%d",
                                       default_value=0, callback=callback_select_grid,
                                       tag='_slider_current_grid')

                with dpg.group(horizontal=True):
                    dpg.add_text("Region Selection and Grid Building: ")

                    def callback_project(sender, app_data):
                        print(self.selected_points)
                        self.project_points = True

                    def callback_grow(sender, app_data):
                        self.grow_reg = True
                        self.need_update = True

                    def callback_show_edit_grid(sender, app_data):
                        if self.show_grid == 'edit':
                            self.show_grid = 'density'
                            dpg.configure_item("_button_show_grid", label="Show Edit Grid")
                        elif self.show_grid == 'grow':
                            self.show_grid = 'edit'
                            dpg.configure_item("_button_show_grow_grid", label="Show Grow Grid")
                        else:
                            self.show_grid = 'edit'
                            dpg.configure_item("_button_show_grid", label="Don't Show Edit Grid")
                        self.need_update = True

                    def callback_reset_grid(sender, app_data):
                        self.current_grid.reset()
                        self.need_update = True

                    dpg.add_button(label="Show Edit Grid", tag="_button_show_grid", callback=callback_show_edit_grid)
                    dpg.bind_item_theme("_button_show_grid", theme_button)

                    dpg.add_button(label="Reset Edit Grid", tag="_button_reset_grid", callback=callback_reset_grid)
                    dpg.bind_item_theme("_button_reset_grid", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text('Grid Logical Operations:')
                    def callback_grid_xor(sender, app_data):
                        self.grid.xor(self.negative_grid.get_grid())
                        self.negative_grid.reset()
                        self.need_update = True

                    def callback_grid_and(sender, app_data):
                        self.grid.and_(self.negative_grid.get_grid())
                        self.negative_grid.reset()
                        self.need_update = True

                    dpg.add_button(label="Grid XOR", tag="_button_grid_XOR", callback=callback_grid_xor)
                    dpg.bind_item_theme("_button_grid_XOR", theme_button)

                    dpg.add_button(label="Grid AND", tag="_button_grid_AND", callback=callback_grid_and)
                    dpg.bind_item_theme("_button_grid_AND", theme_button)

                dpg.add_viewport_drawlist(front=True, tag="viewport_front")
                with dpg.group(horizontal=True):

                    def callback_click_when_selecting(sender, app_data):
                        if self.select_grid_pos:
                            mouse_pos = dpg.get_mouse_pos()
                            self.selected_points.append(mouse_pos)
                    def callback_selection(sender, app_data):
                        self.select_grid_pos = False if self.select_grid_pos else True
                        print(f"Selection is turned {'on' if self.select_grid_pos else 'off'}")

                    with dpg.handler_registry():
                        dpg.add_mouse_click_handler(callback=callback_click_when_selecting)
                        dpg.add_key_press_handler(key=dpg.mvKey_B, callback=callback_selection)

                    dpg.add_button(label="Selection", tag="_button_selection", callback=callback_selection)
                    dpg.bind_item_theme("_button_selection", theme_button)

                    dpg.add_button(label="Project", tag="_button_project", callback=callback_project)
                    dpg.bind_item_theme("_button_project", theme_button)

                    dpg.add_button(label="Grow Region", tag="_button_grow", callback=callback_grow)
                    dpg.bind_item_theme("_button_grow", theme_button)

                def callback_set_growing_steps(sender, app_data):
                    self.growing_steps = app_data

                def callback_set_growing_its(sender, app_data):
                    self.growing_iterations = app_data

                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="Num Growing Iterations", min_value=1, max_value=10000,
                                       default_value=self.growing_iterations, callback=callback_set_growing_its)

                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="Growing Steps", min_value=1, max_value=15, format="%d",
                                       default_value=self.growing_steps, callback=callback_set_growing_steps)

                with dpg.group(horizontal=True):
                    dpg.add_text("Edit Grid Utils")
                with dpg.group(horizontal=True):
                    def callback_nameeditgrid(sender, app_data):
                        self.edit_grid_str = app_data

                    dpg.add_input_text(label="Edit Grid Name", tag="_text_editgrid", callback=callback_nameeditgrid)

                with dpg.group(horizontal=True):
                    def callback_save_grid(sender, app_data):
                        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder,
                                            f'{self.edit_grid_str}.pth')
                        self.grid.save_grid_as_torch(path)

                    def callback_load_edit_grid(sender, app_data):
                        self.grid.load_grid_as_torch(app_data['file_path_name'])

                    with dpg.file_dialog(
                            directory_selector=False, show=False, callback=callback_load_edit_grid,
                            tag="file_dialog_loadeditgrid", width=700, height=400,
                            default_path=os.path.join(os.getcwd())):
                        dpg.add_file_extension("Torch Tensor {.pth}",
                                               color=(0, 255, 0, 255))
                    def callback_load_grid(sender, app_data):
                        dpg.show_item("file_dialog_loadeditgrid")

                    dpg.add_button(label="Save", tag="_button_savegrid", callback=callback_save_grid)
                    dpg.bind_item_theme("_button_savegrid", theme_button)

                    dpg.add_button(label="Load", tag="_button_loadgrid", callback=callback_load_grid)
                    dpg.bind_item_theme("_button_loadgrid", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Growing Grid Utils")
                with dpg.group(horizontal=True):
                    def callback_namegrowgrid(sender, app_data):
                        self.grow_grid_str = app_data

                    dpg.add_input_text(label="Growing Grid Name", tag="_text_growgrid", callback=callback_namegrowgrid)

                with dpg.group(horizontal=True):
                    def callback_savegrowgrid(sender, app_data):
                        path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder,
                                            f'{self.grow_grid_str}.pth')
                        self.growing_grid.save_grid_as_torch(path)

                    def callback_growgrid(sender, app_data):
                        self.growing_grid.grid_from_growing_queue(
                            density_thresh=self.trainer.model.density_thresh,
                            grid=self.grid,
                            density_grid=self.trainer.model.density_grid
                        )

                    dpg.add_button(label="Extract Growing Grid", tag="_button_growgrid", callback=callback_growgrid)
                    dpg.bind_item_theme("_button_growgrid", theme_button)

                    dpg.add_button(label="Save", tag="_button_savegrowgrid", callback=callback_savegrowgrid)
                    dpg.bind_item_theme("_button_savegrowgrid", theme_button)

                    def callback_load_grow_grid(sender, app_data):
                        self.growing_grid.load_grid_as_torch(app_data['file_path_name'])

                    with dpg.file_dialog(
                            directory_selector=False, show=False, callback=callback_load_grow_grid,
                            tag="file_dialog_loadgrowgrid", width=700, height=400,
                            default_path=os.path.join(os.getcwd())):
                        dpg.add_file_extension("Torch Tensor {.pth}",
                                               color=(0, 255, 0, 255))

                    def callback_load_grid(sender, app_data):
                        dpg.show_item("file_dialog_loadgrowgrid")

                    dpg.add_button(label="Load", tag="_button_loadgrowgrid", callback=callback_load_grid)
                    dpg.bind_item_theme("_button_loadgrowgrid", theme_button)

                    def callback_show_grow_grid(sender, app_data):
                        if self.show_grid == 'grow':
                            self.show_grid = 'density'
                            dpg.configure_item("_button_show_grow_grid", label="Show Grow Grid")
                        elif self.show_grid == 'edit':
                            self.show_grid = 'grow'
                            dpg.configure_item("_button_show_grid", label="Show Edit Grid")
                        else:
                            self.show_grid = 'grow'
                            dpg.configure_item("_button_show_grow_grid", label="Don't Show Grow Grid")
                        self.need_update = True

                    dpg.add_button(label="Show Grow Grid", tag="_button_show_grow_grid",
                                   callback=callback_show_grow_grid)
                    dpg.bind_item_theme("_button_show_grow_grid", theme_button)

                    def callback_reset_ggrid(sender, app_data):
                        self.growing_grid.reset()
                        self.need_update = True


                    dpg.add_button(label="Reset Grow Grid", tag="_button_reset_ggrid", callback=callback_reset_ggrid)
                    dpg.bind_item_theme("_button_reset_ggrid", theme_button)

            with dpg.collapsing_header(label="Style Transfer", default_open=True):
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_text("Train StyleEnc: ")

                    def callback_load_style_image(sender, app_data):
                        # were dealing with a png, idk why this is stupid...
                        if app_data["file_name"][-1] == '*':
                            img_name = list(app_data["selections"].keys())[0]
                            img_ending = img_name[img_name.rfind('.')+1:]
                            print(f'{app_data["file_path_name"][:-1]}{img_ending}')
                            style_img = torchvision.io.read_image(f'{app_data["file_path_name"][:-1]}{img_ending}')
                        else:
                            style_img = torchvision.io.read_image(app_data['file_path_name'])
                        # if smaller, reshape
                        if style_img.shape[1] < self.style_img.shape[0] or style_img.shape[2] < self.style_img.shape[1]:
                            resize_ = torchvision.transforms.Resize((self.style_size, self.style_size))
                            self.style_img_original = resize_((style_img) / 255.).permute(1, 2, 0).numpy()
                            self.style_img = self.style_img_original.copy()
                        else:
                            self.style_img_original = (style_img / 255.).permute(1, 2, 0).numpy()
                            self.style_img_scaled = self.style_img_original.copy()
                            self.style_img = self.style_img_original[:self.style_size, :self.style_size].copy()
                            dpg.configure_item("style_img_offset_x",
                                               max_value=self.style_img_original.shape[1] - self.style_size)
                            dpg.configure_item("style_img_offset_y",
                                               max_value=self.style_img_original.shape[0] - self.style_size)
                            dpg.configure_item("style_img_scale",
                                               min_value=max(self.style_size / self.style_img_original.shape[0],
                                                             self.style_size / self.style_img_original.shape[1]))

                        dpg.set_value("_style_texture", self.style_img)
                        dpg.configure_item("_style_window", show=True)

                    with dpg.file_dialog(
                        directory_selector=False, show=False, callback=callback_load_style_image,
                            tag="file_dialog_styleimage", width=700, height=400,
                            default_path=os.path.join(os.getcwd(), 'style_images')):
                        dpg.add_file_extension(".*")
                        dpg.add_file_extension("Images (*.jpg *.jpeg *.png){.jpg, .jpeg, .png}",
                                               color=(0, 255, 0, 255))

                    def callback_train_styleenc(sender, app_data):
                        if self.train_styleenc:
                            self.train_styleenc = False
                            dpg.configure_item("_button_encodestyle", label="start")
                        else:
                            self.train_styleenc = True
                            self.show_styleenc = True
                            dpg.configure_item("_button_encodestyle", label="stop")

                    def callback_select_styleimg(sender, app_data):
                        dpg.show_item("file_dialog_styleimage")

                    dpg.add_button(label="start", tag="_button_encodestyle", callback=callback_train_styleenc)
                    dpg.bind_item_theme("_button_encodestyle", theme_button)

                    dpg.add_button(label="Select Style Image", tag="_button_styleimg", callback=callback_select_styleimg)
                    dpg.bind_item_theme("_button_styleimg", theme_button)

                    def callback_save_styleenc(sender, app_data):
                        if self.style_encoder is not None:
                            path = os.path.join(self.opt.ablation_dir, self.opt.ablation_folder)
                            torch.save(self.style_encoder,
                                       os.path.join(path, "style_enc.pth"))
                            print(f"Saved style encoder to {os.path.join(path)}")

                    dpg.add_button(label="Save", tag="_button_savestyleenc", callback=callback_save_styleenc)
                    dpg.bind_item_theme("_button_savestyleenc", theme_button)

                    def callback_reset_editdataset(sender, app_data):
                        self.edit_dataset = None
                        self.distill_step = 0
                        self.step = 0
                        self.style_encoder = None
                        self.palet = torch.zeros((self.opt.num_palette_bases, 3), dtype=torch.float32)
                        for i in range(self.palet.shape[0], self.opt.num_palette_bases):
                            dpg.configure_item(f"_palette_{i}", show=False)
                        self.original_palet = torch.zeros((self.opt.num_palette_bases, 3), dtype=torch.float32)
                        self.trainer.style_optimizer = None

                    dpg.add_button(label="Reset Dataset", tag="_button_reseteditdset", callback=callback_reset_editdataset)
                    dpg.bind_item_theme("_button_reseteditdset", theme_button)

                    def callback_load_styleenc(sender, app_data):
                        self.style_encoder = torch.load(app_data['file_path_name'])
                        self.show_styleenc = True
                        self.palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()
                        self.original_palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()

                        dpg.configure_item("slider_palette_id_", max_value=self.palet.shape[0] - 1)

                        # set the last palette(s) to invisible
                        if self.palet.shape[0] < self.opt.num_palette_bases:
                            for i in range(self.palet.shape[0], self.opt.num_palette_bases):
                                dpg.configure_item(f"_palette_{i}", show=False)

                        dpg.show_item("_text_distill")
                        dpg.show_item("_button_distill")
                        self.refresh_palet()

                    with dpg.file_dialog(
                        directory_selector=False, show=False, callback=callback_load_styleenc,
                            tag="file_dialog_loadstyleenc", width=700, height=400, default_path=os.getcwd()):
                        dpg.add_file_extension("StyleEncoder (*.pth){.pth}",
                                               color=(0, 255, 0, 255))

                    def callback_load_styleenc_dialog(sender, app_data):
                        dpg.show_item("file_dialog_loadstyleenc")

                    dpg.add_button(label="Load", tag="_button_loadstyleenc", callback=callback_load_styleenc_dialog)
                    dpg.bind_item_theme("_button_loadstyleenc", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Distill: ", show=False, tag="_text_distill")
                    def callback_edit(sender, app_data):
                        if self.distill:
                            self.distill = False
                            dpg.configure_item("_button_distill", label="start")
                        else:
                            self.distill = True
                            self.show_styleenc = False
                            self.need_update = True
                            self.trainfast = True
                            self.trainer.lr_scheduler = optim.lr_scheduler.LambdaLR(self.trainer.optimizer, lambda iter: 0.1 ** min(iter / self.opt.train_steps_distill, 1))
                            dpg.configure_item("_button_distill", label="stop")

                    dpg.add_button(label="start", tag="_button_distill", callback=callback_edit, show=False)
                    dpg.bind_item_theme("_button_distill", theme_button)

                    def callback_train_fast(sender, app_data):
                        self.trainfast = app_data
                    dpg.add_checkbox(label="No Updates:", default_value=self.trainfast, callback=callback_train_fast)

                def callback_set_palet(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.palet[self.highlight_palette_id, user_data[1]] = app_data
                    self.sync_with_styleenc()

                palet_width = 125
                palet_min_max = [-1, 2]

                def callback_set_palette_id(sender, app_data):
                    self.highlight_palette_id = app_data
                    self.refresh_palet()

                dpg.add_slider_int(label="Palette_ID", min_value=0, max_value=self.opt.num_palette_bases - 1, format="%d",
                                   default_value=self.highlight_palette_id, callback=callback_set_palette_id,
                                   tag='slider_palette_id_')

                def callback_baryweights(sender, app_data):
                    self.show_baryweights = not self.show_baryweights
                    if self.show_baryweights:
                        self.show_offsets = False
                        dpg.set_value('_checkbox_offsets', False)
                    self.need_update = True

                def callback_offsets(sender, app_data):
                    self.show_offsets = not self.show_offsets
                    if self.show_offsets:
                        self.show_baryweights = False
                        dpg.set_value('_checkbox_weights', False)
                    self.need_update = True

                def callback_use_offsets(sender, app_data):
                    self.use_offsets = not self.use_offsets
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Show Weights", tag="_checkbox_weights",
                                     default_value=self.show_baryweights, callback=callback_baryweights)
                    dpg.add_checkbox(label="Show Offsets", tag="_checkbox_offsets",
                                     default_value=self.show_offsets, callback=callback_offsets)
                    dpg.add_checkbox(label="Use Offsets", tag="_checkbox_use_offsets",
                                     default_value=self.use_offsets, callback=callback_use_offsets)

                def callback_change_palette_(sender, app_data):
                    self.palet[self.highlight_palette_id] = torch.tensor(app_data[:3], dtype=torch.float32)
                    self.refresh_palet()

                def callback_change_highlight(sender, app_data, user_data):
                    self.highlight_palette_id = user_data
                    dpg.set_value('slider_palette_id_', self.highlight_palette_id)
                    self.refresh_palet()

                with dpg.group(horizontal=True):
                    for i in range(self.palet.shape[0]):
                        dpg.add_color_edit(default_value=list((self.palet[i] * 255).byte().numpy()), no_inputs=True,
                                           label="", width=200, tag=f"_palette_{i}", no_alpha=True, no_picker=True,
                                           user_data=i, callback=callback_change_highlight, no_tooltip=True)
                    def callback_reset_palet(sender, app_data):
                        self.palet = self.original_palet.clone()
                        self.refresh_palet()

                    dpg.add_button(label="Reset", tag="_button_reset_palet", callback=callback_reset_palet)
                    dpg.bind_item_theme("_button_reset_palet", theme_button)

                    def callback_save_palet(sender, app_data):
                        torch.save(self.palet,
                                   os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "palette.pth"))
                        ic(f'palette {self.palet} saved')

                    dpg.add_button(label="Save", tag="_button_save_palet", callback=callback_save_palet)
                    dpg.bind_item_theme("_button_save_palet", theme_button)

                    def callback_load_p(sender, app_data):
                        self.palet = torch.load(app_data['file_path_name'])
                        self.refresh_palet()
                    def callback_load_palet(sender, app_data):
                        dpg.show_item("file_dialog_palette")

                    with dpg.file_dialog(
                            directory_selector=False, show=False, callback=callback_load_p,
                            tag="file_dialog_palette", width=700, height=400,
                            default_path=os.path.join(os.getcwd(), self.opt.ablation_dir, self.opt.ablation_folder)):
                        dpg.add_file_extension("Torch Tensor {.pth}",
                                               color=(0, 255, 0, 255))

                    dpg.add_button(label="Load", tag="_button_load_palet", callback=callback_load_palet)
                    dpg.bind_item_theme("_button_load_palet", theme_button)

                with dpg.group(horizontal=True):
                    def callback_show_palet_window(sender, app_data):
                        dpg.configure_item(f"_palette_window", show=True)

                    dpg.add_button(label='Modify Palette', show=False, tag='_button_modify_palet',
                                   callback=callback_show_palet_window)
                    dpg.bind_item_theme("_button_modify_palet", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text(f"Palette {self.highlight_palette_id}:", tag='palette_number_')
                    dpg.add_color_edit(default_value=list((self.palet[self.highlight_palette_id]* 255).byte().numpy()),
                                       label="Palette Color", width=200, tag="_palette_color_editor", no_alpha=True,
                                       callback=callback_change_palette_)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="R", width=palet_width, min_value=palet_min_max[0], max_value=palet_min_max[1],
                                         format="%.2f", default_value=self.palet[0,0], tag='p0',
                                         callback=callback_set_palet, user_data=[0,0])

                    dpg.add_slider_float(label="G", width=palet_width, min_value=palet_min_max[0], max_value=palet_min_max[1],
                                         format="%.2f", default_value=self.palet[0,1], tag='p1',
                                         callback=callback_set_palet, user_data=[0,1])

                    dpg.add_slider_float(label="B", width=palet_width, min_value=palet_min_max[0], max_value=palet_min_max[1],
                                         format="%.2f", default_value=self.palet[0,2], tag='p2',
                                         callback=callback_set_palet, user_data=[0,2])

            with dpg.collapsing_header(label="Reference-Based Stylization", default_open=False):

                with dpg.group(horizontal=True):
                    dpg.add_text("Train StyleEnc: ")

                    def callback_load_path_npr(sender, app_data):
                        self.npr_string = app_data['file_path_name']

                    dpg.add_file_dialog(
                            directory_selector=True, show=False, callback=callback_load_path_npr,
                            tag="file_dialog_npr_config", width=700, height=400,
                            default_path=os.path.join(os.getcwd(), 'single_view_stylization'))

                    def callback_train_styleenc_npr(sender, app_data):
                        if self.train_styleenc:
                            self.train_styleenc_npr = False
                            dpg.configure_item("_button_npr", label="start")
                        else:
                            self.train_styleenc_npr = True
                            self.show_styleenc = True
                            dpg.configure_item("_button_npr", label="stop")

                    def callback_select_npr_config(sender, app_data):
                        dpg.show_item("file_dialog_npr_config")

                    dpg.add_button(label="start", tag="_button_npr", callback=callback_train_styleenc_npr)
                    dpg.bind_item_theme("_button_npr", theme_button)

                    dpg.add_button(label="Select Config", tag="_button_npr_config",
                                   callback=callback_select_npr_config)
                    dpg.bind_item_theme("_button_npr_config", theme_button)

                    def callback_npr_distill(sender, app_data):
                        if self.distill_npr:
                            self.distill_npr = False
                            self.training = False
                        else:
                            self.training = True
                            self.distill_npr = True
                            self.show_styleenc = False
                            self.need_update = True

                    dpg.add_button(label="Distill", tag="_button_npr_distill", callback=callback_npr_distill)
                    dpg.bind_item_theme("_button_npr_distill", theme_button)

            with dpg.collapsing_header(label="Utils", default_open=False):

                with dpg.group(horizontal=True):
                    def callback_val_styleenc(sender, app_data):
                        self.eval = True
                        self.eval_type = 'styleenc_val'

                    dpg.add_button(label="Val Style Predictor", tag="_button_val_styleenc",
                                   callback=callback_val_styleenc)
                    dpg.bind_item_theme("_button_val_styleenc", theme_button)

                    def callback_test_styleenc(sender, app_data):
                        self.eval = True
                        self.eval_type = 'styleenc_test'

                    dpg.add_button(label="Test Style Predictor", tag="_button_test_styleenc",
                                   callback=callback_test_styleenc)
                    dpg.bind_item_theme("_button_test_styleenc", theme_button)

                with dpg.group(horizontal=True):
                    def callback_render_val(sender, app_data):
                        self.eval = True
                        self.eval_type = 'val'

                    dpg.add_button(label="Render Val", tag="_button_eval_nerf", callback=callback_render_val)
                    dpg.bind_item_theme("_button_eval_nerf", theme_button)

                    def callback_render_train(sender, app_data):
                        self.eval = True
                        self.eval_type = 'train'

                    dpg.add_button(label="Render Train", tag="_button_render_train", callback=callback_render_train)
                    dpg.bind_item_theme("_button_render_train", theme_button)

                    def callback_render_test(sender, app_data):
                        self.eval = True
                        self.eval_type = 'test'

                    dpg.add_button(label="Render Test", tag="_button_render_test", callback=callback_render_test)
                    dpg.bind_item_theme("_button_render_test", theme_button)
                    def callback_render_video(sender, app_data):
                        self.eval = True
                        self.eval_type = 'video'

                    dpg.add_button(label="Render Video", tag="_button_render_vid", callback=callback_render_video)
                    dpg.bind_item_theme("_button_render_vid", theme_button)

                with dpg.group(horizontal=True):
                    def callback_eval_mask(sender, app_data):
                        self.eval = True
                        self.eval_type = 'mask'

                    dpg.add_button(label="Eval with masks", tag="_button_eval_mask", callback=callback_eval_mask)
                    dpg.bind_item_theme("_button_eval_mask", theme_button)

                    def callback_screenshot(sender, app_data):
                        self.screenshot()

                    dpg.add_button(label="Screenshot", tag="_button_screenshot", callback=callback_screenshot)
                    dpg.bind_item_theme("_button_screenshot", theme_button)

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=False):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode,
                              callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor",
                                   no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f",
                                     default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d",
                                   default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    # self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound,
                                         callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound,
                                         callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound,
                                         callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound,
                                         callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0,
                                         format="%.2f", default_value=-self.opt.bound,
                                         callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound,
                                         format="%.2f", default_value=self.opt.bound,
                                         callback=callback_set_aabb, user_data=5)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        with dpg.window(tag="_style_window", width=self.style_size + 100, height=self.style_size + 100,
                        pos=(0, 0), show=False):
            # add the texture
            dpg.add_image("_style_texture")

            def callback_change_offset(sender, app_data, user_data):
                self.style_img_offsets[user_data] = app_data
                img_ = self.style_img_scaled[
                       self.style_img_offsets[0]:self.style_img_offsets[0] + self.style_size,
                       self.style_img_offsets[1]:self.style_img_offsets[1] + self.style_size]
                self.style_img = img_.copy()
                dpg.set_value("_style_texture", self.style_img)

            def callback_change_scale(sender, app_data):
                # recalculate the scale
                dpg.configure_item("style_img_offset_x",
                                   max_value=self.style_img_scaled.shape[1] - self.style_size)
                dpg.configure_item("style_img_offset_y",
                                   max_value=self.style_img_scaled.shape[0] - self.style_size)
                dpg.set_value("style_img_offset_y", 0)
                dpg.set_value("style_img_offset_x", 0)
                self.style_img_offsets[:] = 0

                scale = app_data
                resize_ = torchvision.transforms.Resize([int(np.ceil(self.style_img_original.shape[0] * scale)),
                                                        int(np.ceil(self.style_img_original.shape[1] * scale))])

                img_ = resize_(torch.from_numpy(self.style_img_original).permute(-1, 0, 1))
                self.style_img_scaled = img_.permute(1,2,0).numpy().copy()
                self.style_img = self.style_img_scaled[:self.style_size, :self.style_size].copy()
                dpg.set_value("_style_texture", self.style_img)

            dpg.add_slider_int(label="Offset x", tag='style_img_offset_x', user_data=1, callback=callback_change_offset)
            dpg.add_slider_int(label="Offset y", tag='style_img_offset_y', user_data=0, callback=callback_change_offset)
            dpg.add_slider_float(label="Scale", tag='style_img_scale', callback=callback_change_scale, max_value=2,
                                 default_value=1)

            def callback_setstyle(sender, app_data, user_data):
                # close the current window
                dpg.configure_item("_style_window", show=False)
                self.style_img_set = True

            dpg.add_button(label="Set Style Image", tag="_button_set_style_img", callback=callback_setstyle)
            dpg.bind_item_theme("_button_set_style_img", theme_button)

        with dpg.window(tag="_palette_window", width=400, height=400,
                        pos=(0, 0), show=False):
            def callback_change_weights(sender, app_data, user_data):
                self.palet_weights[int(user_data)] = app_data
                self.need_update = True
            def callback_change_bias(sender, app_data, user_data):
                self.palet_biases[int(user_data)] = app_data
                self.need_update = True

            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Weight", tag='txt_weight_idk', indent=100)
                dpg.add_text(default_value="Bias", tag='txt_bias_idk', indent=300)

            for i in range(self.opt.num_palette_bases):
                with dpg.group(horizontal=True):
                    dpg.add_color_edit(default_value=list((self.palet[i] * 255).byte().numpy()), no_inputs=True,
                                       label="", width=50, tag=f"_palette_modify_{i}", no_alpha=True, no_picker=True,
                                       user_data=i, no_tooltip=True)
                    dpg.add_slider_float(label="", tag=f'palette_weight_{i}', user_data=i,
                                         min_value=0, max_value=10, default_value=self.palet_weights[i],
                                         callback=callback_change_weights, width=200)
                    dpg.add_slider_float(label="", tag=f'palette_bias_{i}', user_data=i, min_value=-1,
                                         max_value=1, callback=callback_change_bias, width=200,
                                         default_value=self.palet_biases[i])

            def callback_palet_reset(sender, app_data):
                self.palet_biases[:] = 0.
                self.palet_weights[:] = 1.
                self.need_update = True

                for i in range(self.opt.num_palette_bases):
                    dpg.set_value(f'palette_weight_{i}', self.palet_weights[i])
                    dpg.set_value(f'palette_bias_{i}', self.palet_biases[i])

            def callback_palet_close(sender, app_data):
                dpg.configure_item('_palette_window', show=False)

            with dpg.group(horizontal=True):
                dpg.add_button(label='reset', tag='btn_palette_weight_reset', callback=callback_palet_reset)
                dpg.add_button(label='Close', tag='btn_palette_close', callback=callback_palet_close)

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        # add a font registry
        with dpg.font_registry():
            # first argument ids the path to the .ttf or .otf file
            default_font = dpg.add_font("Ubuntu-Light.ttf", 16)
            dpg.bind_font(default_font)


        dpg.create_viewport(title='torch-ngp', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.bind_item_theme("_style_window", theme_no_padding)
        dpg.bind_item_theme("_palette_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()

    def refresh_palet(self):
        dpg.set_value('p0', self.palet[self.highlight_palette_id, 0])
        dpg.set_value('p1', self.palet[self.highlight_palette_id, 1])
        dpg.set_value('p2', self.palet[self.highlight_palette_id, 2])
        dpg.set_value('_palette_color_editor', list((self.palet[self.highlight_palette_id] * 255).byte().numpy()))

        for i in range(self.palet.shape[0]):
            dpg.set_value(f'_palette_{i}', list((self.palet[i] * 255).byte().numpy()))
            dpg.set_value(f'_palette_modify_{i}', list((self.palet[i] * 255).byte().numpy()))

        dpg.set_value('palette_number_', f'Palette: {self.highlight_palette_id}')
        self.sync_with_styleenc()

    def sync_with_styleenc(self):
        if self.style_encoder is not None and not self.train_styleenc:
            self.style_encoder.set_color_palette(self.palet.cuda())
        self.need_update = True

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                if (self.distill or self.distill_npr) and self.step == 0:
                    self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True)
                    self.starter.record()
                self.train_styleenc = False
                self.train_step()
                if (self.distill_npr or self.distill) and self.step > self.opt.train_steps_distill:
                    self.training = False
                    self.distill = False
                    self.trainfast = False
                    self.need_update = True

                    self.ender.record()
                    torch.cuda.synchronize()
                    t = self.starter.elapsed_time(self.ender)
                    self.timings.append(t / 1000.)

                    self.trainer.save_checkpoint(path=os.path.join(self.opt.ablation_dir, self.opt.ablation_folder),
                                                 full=True)
                    #self.eval_masked(loader1=self.val_loader, prefix='masked', loader2=self.test_loader,
                    #                 loader3=self.train_loader)
                    self.eval_nerf(loader=self.val_loader, prefix='val')
                    self.eval_nerf(loader=self.test_loader, prefix='test')
                    self.eval_nerf(loader=self.video_loader, prefix='video')
                    #self.eval_nerf(loader=self.train_loader, prefix='train')

                    # save original and modified palette
                    torch.save(self.original_palet,
                               os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "palet_og.pth"))
                    torch.save(self.palet,
                               os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "palet_mod.pth"))

                    if len(self.timings) > 3:
                        timings_ = {
                            "edit_dataset": f'{self.timings[0]:.2f} s',
                            "train_style_enc": f'{self.timings[1]:.2f} s',
                            "distill_dataset": f'{self.timings[2]:.2f} s',
                            "distill_nerf": f'{self.timings[-1]:.2f} s',
                            "sum": f'{(self.timings[-2] + self.timings[-1]):.2f} s',
                        }

                        with open(os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "timings.json"),
                                  "w") as outfile:
                            json.dump(timings_, outfile, indent=2)

                    if self.opt.run_all:
                        exit(0)
            if self.distill and not self.training:
                dpg.configure_item("_button_train", label="stop")
                if self.edit_dataset is None:
                    self.edit_dataset = EditDataset(self.opt, self.train_loader, self.grid, self.growing_grid, self.trainer, depth_diff=self.opt.depth_diff).dataloader()
                self.distill_dataset(self.train_loader,
                                     self.style_encoder,
                                     self.edit_dataset,
                                     save_train_dataset=True)
                self.training = True
            if self.train_styleenc or self.train_styleenc_npr:
                self.need_update = False
                if self.distill_step < self.opt.train_steps_style:
                    if self.style_encoder is None:
                        print("started training the Style Encoder")
                        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        self.starter.record()
                        style_img = torch.from_numpy(self.style_img).permute(-1, 0, 1) if self.style_img_set else None
                        self.init_style_pred(style_img)
                    if self.distill_step > self.opt.train_steps_style - self.opt.distill_palette_steps:
                        self.style_encoder.distill_color_palettes(self.edit_dataset)
                        self.opt.distill_palette_steps = -1
                        # update the gui
                        self.original_palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()
                        self.palet = self.style_encoder.get_color_palette().cpu().detach().float().clone()
                        dpg.configure_item("slider_palette_id_", max_value=self.palet.shape[0] - 1)
                        dpg.configure_item("_button_modify_palet", show=True)

                        # set the last palette(s) to invisible
                        if self.palet.shape[0] < self.opt.num_palette_bases:
                            for i in range(self.palet.shape[0], self.opt.num_palette_bases):
                                dpg.configure_item(f"_palette_{i}", show=False)
                                dpg.configure_item(f'_palette_modify_{i}', show=False)
                                dpg.configure_item(f'palette_weight_{i}', show=False)
                                dpg.configure_item(f'palette_bias_{i}', show=False)

                        self.refresh_palet()
                    if self.train_styleenc_npr:
                        self.train_style_step_npr()
                    else:
                        self.train_style_step()
                    self.palet = self.style_encoder.get_color_palette().cpu().detach().float()
                    self.original_palet = self.style_encoder.get_color_palette().cpu().detach().float()
                    self.refresh_palet()
                    if not self.trainfast:
                        self.need_update = True
                else:

                    self.ender.record()
                    torch.cuda.synchronize()
                    t = self.starter.elapsed_time(self.ender)
                    self.timings.append(t / 1000.)

                    dpg.show_item("_text_distill")
                    dpg.show_item("_button_distill")
                    #self.eval_style_predictor(loader=self.val_loader, folder_name='val_styleenc')
                    if self.opt.run_all:
                        #self.starter.record()
                        if self.train_styleenc_npr:
                            self.distill_npr = True
                        else:
                            self.distill = True
                            self.trainfast = True
                        self.show_styleenc = False
                    torch.save(self.style_encoder,
                               os.path.join(self.opt.ablation_dir, self.opt.ablation_folder, "style_enc.pth"))
                    self.train_styleenc = False
                    self.train_styleenc_npr = False
                    dpg.configure_item("_button_encodestyle", label="stop")

            if self.project_points:
                self.project_points_()
                self.project_points = False
                self.select_grid_pos = False
                self.need_update = True
                self.selected_points = []

            if self.grow_reg:
                for _ in range(self.growing_steps):
                    self.grow_region()
                if self.current_grid == self.growing_grid:
                    self.growing_grid.xor(self.grid.get_grid())
                self.grow_reg = False
                self.need_update = True

            for i in self.selected_points:
                # debug marking
                ms = 6
                self.render_buffer[int(i[-1]) - ms: int(i[-1]) + ms,
                int(i[0]) - ms: int(i[0]) + ms, :] = np.array((1, 0, 0))

            if self.eval:
                if self.eval_type == 'train':
                    self.eval_nerf(loader=self.train_loader, prefix='train')
                elif self.eval_type == 'val':
                    self.eval_nerf(loader=self.val_loader, prefix='val')
                elif self.eval_type == 'video':
                    self.eval_nerf(loader=self.video_loader, prefix='video')
                elif self.eval_type == 'mask':
                    self.eval_masked(loader1=self.val_loader, prefix='masked', loader2=self.test_loader,
                                     loader3=self.train_loader)
                elif self.eval_type == 'test':
                    self.eval_nerf(loader=self.test_loader, prefix='test')
                elif self.eval_type == 'styleenc_val':
                    if self.style_encoder is None:
                        print('no style encoder found')
                    else:
                        self.eval_style_predictor(loader=self.val_loader, folder_name='val_styleenc')
                elif self.eval_type == 'styleenc_test':
                    if self.style_encoder is None:
                        print('no style encoder found')
                    else:
                        self.eval_style_predictor(loader=self.test_loader, folder_name='test_styleenc')
                self.eval = False

            if self.show_styleenc:
                self.test_step_styleenc()
            else:
                if not self.opt.run_all:
                    self.test_step()
            dpg.render_dearpygui_frame()