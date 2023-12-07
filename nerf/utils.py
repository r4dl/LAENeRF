import os
import glob

import torchvision.transforms
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision.io import write_png
import torchvision.transforms as tt
from configargparse import Namespace

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips
from torchmetrics.functional import structural_similarity_index_measure

from editing.style_encoder import StyleEncoder
from editing.semantic_encoder import SemanticEncoder

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, perturb_ray_dirs: bool = False):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    # in Instant-NGP
    # (uv.x() - screen_center.x()) * (float)resolution.x() / focal_length.x(),
    # (uv.y() - screen_center.y()) * (float)resolution.y() / focal_length.y()
    if perturb_ray_dirs:
        offset = (torch.rand(2) - 0.5).to(device)
        xs = ((i - offset[0]) - cx) / fx * zs
        ys = ((j - offset[1]) - cy) / fy * zs
    else:
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # optionally use LPIPS loss for patch-based training
        if self.opt.patch_size > 1:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        self.style_optimizer = None
        self.style_step = 0

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step_npr(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        torch.autograd.set_detect_anomaly(True)

        images = data['target'] # [B, N, 3 or 4]
        # style loss
        style_img = data['style_img']

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if (C == 3 or self.model.bg_radius > 0) and not self.opt.no_bg:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            gt_style = style_img[..., :3] * style_img[..., 3:] + bg_color * (1 - style_img[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color.flatten(0,1),
                                    perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True,
                                    distill=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(gt_rgb.shape)

        # MSE loss
        weights = data['target_weights']
        loss = torch.pow(weights * (pred_rgb - gt_rgb), 2).mean()  # [B, N, 3] --> [B, N]

        loss_style = torch.pow((1 - weights / 2) * (gt_style - pred_rgb), 2).mean()
        loss += self.opt.style_weight_d * loss_style
        # depth loss
        if outputs['depth'].sum() > 0:
            depth_loss = torch.pow(data['depth_weights'].flatten() *
                                   (outputs['depth'].flatten() - (data['depth'].flatten().cuda() - outputs['nears'])), 2).mean()
            loss += self.opt.depth_weight_d * depth_loss

        return pred_rgb, gt_rgb, loss

    def train_step(self, data, distill: bool = False, depth_sup: bool = False, edit_dataset=None):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        torch.autograd.set_detect_anomaly(True)

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if (C == 3 or self.model.bg_radius > 0) and not self.opt.no_bg:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, distill=distill, **vars(self.opt))
        # outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=True, **vars(self.opt))
    
        pred_rgb = outputs['image']

        # depth loss
        depth_loss = None
        if distill and depth_sup:
            depth_weights = torch.zeros_like(outputs['depth'])
            depth_weights[data['depth'].cuda() > 0] = 1
            depth_loss = torch.pow((outputs['depth'] - (data['depth'].cuda() - outputs['nears'])) * depth_weights, 2).mean()

        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        # patch-based rendering
        if self.opt.patch_size > 1:
            gt_rgb = gt_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb = pred_rgb.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            # torch_vis_2d(gt_rgb[0])
            # torch_vis_2d(pred_rgb[0])

            # LPIPS loss [not useful...]
            loss = loss + 1e-3 * self.criterion_lpips(pred_rgb, gt_rgb)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()
        if depth_loss is not None:
            loss += 1e-3 * depth_loss

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss

    def distill_step(self, data, edit_bitfield: torch.cuda.ByteTensor, perturb_depth: bool, grow_grid: bool = False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        if 'images' in data:
            images = data['images']  # [B, N, 3/4]

            if self.opt.color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])

            bg_color = torch.ones_like(images[..., :3]) # [3], per-image random.
        else:
            bg_color=None

        outputs = self.model.run_cuda_distill(rays_o, rays_d, edit_bitfield, bg_color=bg_color, perturb=True,
                                    force_all_rays=False if self.opt.patch_size == 1 else True,
                                    perturb_depth=perturb_depth, grow_grid=grow_grid,
                                    **vars(self.opt))

        pred_rgb = outputs['image']
        pred_term = outputs['x_term']
        pred_w8s = outputs['weights']
        pred_w8s_edit = outputs['weights_edit']
        pred_depth = outputs['depth']
        min_near = outputs['min_near']
        pred_depth_edit = outputs['depth_edit']

        return pred_rgb.squeeze(), pred_term, pred_w8s, pred_w8s_edit, pred_depth.squeeze(), min_near, pred_depth_edit

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt), scale_depth=False)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, edit_grid=None, scale_depth=True):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt),
                                    edit_grid=edit_grid, scale_depth=scale_depth)

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        if len(outputs) > 2:
            return pred_rgb, pred_depth, outputs['t']
        else:
            return pred_rgb, pred_depth


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_gui_npr(self, train_loader, distill=False, step=16, dataloader=None):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(dataloader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            # TODO: do we need to update the occ.grid?
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step_npr(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, distill=False, depth_sup=False, step=16, edit_dataset=None):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data, distill=distill, depth_sup=depth_sup, edit_dataset=edit_dataset)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs


    def train_styleenc_step(self,
                            train_loader,
                            edit_dataset,
                            style_encoder: StyleEncoder,
                            params: Namespace,
                            step: int = 16,
                            global_step: int = 0):

        style_encoder.train()
        loader = iter(train_loader)
        edit_loader = iter(edit_dataset)
        data = next(loader)

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        loss_fct = nn.MSELoss()

        self.model.eval()
        # na\"ive Adam
        if self.style_optimizer is None:
            # naive adam
            self.style_optimizer = optim.Adam(style_encoder.get_params(0.001))

        # preserve_colors
        if params.preserve_color:
            style_encoder.color_transfer_style_img(next(edit_loader)['target'].T)
        #if global_step > (params.train_steps_style - params.warmup_iterations):
        #    self.style_optimizer = optim.Adam(style_encoder.get_params_but_dont_learn_palette(0.001))

        for _ in range(step):
            # random pose for each iteration
            d = next(edit_loader)

            self.style_step += 1

            self.style_optimizer.zero_grad()

            pred_colors, pred_weight, pred_offset = style_encoder.forward_train(
                x=d["x_term"],
                d=d['d'].squeeze()
            )
            #pred_colors = d["w8s"][..., None].half()

            loss = loss_fct(input=pred_colors, target=d["target"].half())
            loss += style_encoder.weights_loss(pred_weight, params).half()
            loss += style_encoder.offset_loss(pred_offset, params).half()
            loss += style_encoder.palet_loss(params).half()

            if (params.style_weight > 0 or params.tv_weight > 0) and global_step > self.opt.warmup_iterations:
                img_to_stylize = torch.zeros((data['H'], data['W'], 3), dtype=torch.float32).cuda()
                img_to_stylize.flatten(0, 1)[d["indices"]] = pred_colors.float()
                img_to_stylize = img_to_stylize[d['minmax'][0].item():d['minmax'][1].item(),
                                                d['minmax'][2].item():d['minmax'][3].item()]
                if params.style_weight > 0:
                    style_loss = style_encoder.style_loss(img_to_stylize.permute(-1, 0, 1))
                    loss += style_loss.half() * params.style_weight

                depth_w_var = d['depth_h_var']
                depth_v_var = d['depth_v_var']

                if params.tv_weight > 0:
                    if params.tv_depth_guide:
                        tv_loss = style_encoder.tv_loss_depth_weighted(img_to_stylize.permute(-1, 0, 1),
                                                                       depth_v_var=depth_v_var, depth_w_var=depth_w_var,
                                                                       weights_trans=d['cut_smooth'])
                    else:
                        tv_loss = style_encoder.tv_loss(img_to_stylize.permute(-1, 0, 1))
                    loss += tv_loss.half() * params.tv_weight

                if params.smooth_trans_weight > 0:
                    smooth_trans_loss = style_encoder.smooth_transition_loss(d['cut_gt'], img_to_stylize,
                                                                             d['cut_smooth'])
                    loss += smooth_trans_loss.half() * params.smooth_trans_weight

                if params.depth_disc_weight > 0:
                    depth_disc_loss = style_encoder.depth_discontinuity_loss(img_to_stylize.permute(-1, 0, 1),
                                                                   depth_v_var=depth_v_var, depth_w_var=depth_w_var)
                    loss += depth_disc_loss.half() * params.depth_disc_weight

            if params.intensity_weight > 0:
                loss += params.intensity_weight * style_encoder.intensity_loss(pred_colors, d["target"].half())

            total_loss += loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.style_optimizer)
            self.scaler.update()

            #if self.scheduler_update_every_step:
            #    self.lr_scheduler.step()


        #if self.ema is not None:
        #    self.ema.update()

        average_loss = total_loss.item() / step

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'its': self.style_step,
            'PSNR': 10 * np.log10(1/average_loss)
        }
        print(f'average loss: {average_loss:.4f}, global step: {self.style_step}')

        return outputs

    def train_styleenc_step_npr(self,
                                train_loader,
                                edit_dataset,
                                style_encoder: StyleEncoder,
                                params: Namespace,
                                semantic_encoder: SemanticEncoder,
                                step: int = 16,
                                global_step: int = 0):

        style_encoder.train()
        loader = iter(train_loader)
        edit_loader = iter(edit_dataset)
        data = next(loader)

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        loss_fct = nn.MSELoss()

        self.model.eval()
        # na\"ive Adam
        if self.style_optimizer is None:
            # naive adam
            self.style_optimizer = optim.Adam(style_encoder.get_params(0.001))

        # preserve_color?
        if params.preserve_color:
            style_encoder.color_transfer_style_img(next(edit_loader)['target'].T)

        for _ in range(step):
            # random pose for each iteration
            d = next(edit_loader)

            self.style_step += 1

            self.style_optimizer.zero_grad()

            pred_colors, pred_weight, pred_offset = style_encoder.forward_train(
                x=d["x_term"],
                d=d['d'].squeeze()
            )
            pred_colors *= d["w8s"][..., None].half()

            if 'target_feat' in d:
                if 'target_weights' in d:
                    loss = (torch.pow(d['target'] - pred_colors[d["indices"]], 2) * d['target_weights'][..., None]).mean() * self.opt.mse_loss
                else:
                    loss = loss_fct(input=pred_colors[d["indices"]], target=d['target'].half()) * self.opt.mse_loss
            else:
                loss = loss_fct(input=pred_colors, target=d["target"].half())
            loss += style_encoder.weights_loss(pred_weight, params).half()
            loss += style_encoder.offset_loss(pred_offset, params).half()
            loss += style_encoder.palet_loss(params).half()

            if 'target_feat' in d:
                img_to_stylize = torch.zeros((data['H'], data['W'], 3), dtype=torch.float32).cuda()
                img_to_stylize.flatten(0, 1)[d["indices__"]] = pred_colors.float()
                img_to_stylize_r = img_to_stylize[d['minmax'][0].item():d['minmax'][1].item(),
                                                d['minmax'][2].item():d['minmax'][3].item()]

                actual_feat = semantic_encoder.encode_feats(img_to_stylize_r.permute(-1, 0, 1),
                                                            layers=[11, 13, 15],
                                                            size=(self.opt.feature_size, self.opt.feature_size))
                loss += semantic_encoder.cos_loss(actual_feat.flatten(-2, -1), d['target_feat']) * self.opt.cos_loss_factor

                color_pred = semantic_encoder.get_mean_patch_color(img_to_stylize.permute(-1, 0, 1),
                                                                   size=(d["color_patch"].shape[-2], d["color_patch"].shape[-1]))
                loss += loss_fct(color_pred, d["color_patch"]) * self.opt.color_patch_loss

                if params.style_weight > 0:
                    style_loss = style_encoder.style_loss(img_to_stylize_r.permute(-1, 0, 1), guide=d['style_guide'])
                    loss += style_loss.half() * params.style_weight

                depth_w_var = d['depth_h_var']
                depth_v_var = d['depth_v_var']

                if params.tv_weight > 0:
                    if params.tv_depth_guide:
                        tv_loss = style_encoder.tv_loss_depth_weighted(img_to_stylize_r.permute(-1, 0, 1),
                                                                       depth_v_var=depth_v_var, depth_w_var=depth_w_var,
                                                                       weights_trans=1 - d['style_guide'])
                    else:
                        tv_loss = style_encoder.tv_loss(img_to_stylize_r.permute(-1, 0, 1))
                    loss += tv_loss.half() * params.tv_weight

                if params.smooth_trans_weight > 0:
                    smooth_trans_loss = style_encoder.smooth_transition_loss(d['cut_gt'], img_to_stylize_r,
                                                                             d['cut_smooth'])
                    loss += smooth_trans_loss.half() * params.smooth_trans_weight

                if params.depth_disc_weight > 0:
                    depth_disc_loss = style_encoder.depth_discontinuity_loss(img_to_stylize_r.permute(-1, 0, 1),
                                                                   depth_v_var=depth_v_var, depth_w_var=depth_w_var)
                    loss += depth_disc_loss.half() * params.depth_disc_weight

            if params.intensity_weight > 0:
                loss += params.intensity_weight * style_encoder.intensity_loss(pred_colors, d["target"].half())

            total_loss += loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.style_optimizer)
            self.scaler.update()

            #if self.scheduler_update_every_step:
            #    self.lr_scheduler.step()


        #if self.ema is not None:
        #    self.ema.update()

        average_loss = total_loss.item() / step

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'its': self.style_step,
            'PSNR': 10 * np.log10(1/average_loss)
        }
        print(f'average loss: {average_loss:.4f}, global step: {self.style_step}')

        return outputs

    def project_points(self, pose, intrinsics, W, H, selected_points, downscale=1):

        if len(selected_points) == 0:
            return

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        xs, ys = tuple(round(p[1] * downscale) for p in selected_points), tuple(round(p[0] * downscale) for p in selected_points)

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, _ = self.test_step(data, perturb=False, scale_depth=False)

        if self.ema is not None:
            self.ema.restore()

        d = rays['rays_d'].squeeze().reshape(rH, rW, -1)[xs, ys]
        o = rays['rays_o'].squeeze().reshape(rH, rW, -1)[xs, ys]
        x_term = o + preds_depth[:, xs, ys].permute(-1, 0) * d

        outputs = {
            'term': x_term,
        }

        return outputs

    # [GUI] test on a single image
    def test_gui_styleenc(self, pose, intrinsics, W, H, edit_grid, style_enc: StyleEncoder, bg_color=None, spp=1,
                          downscale=1, show_index=-1, show_weights=True, use_offsets=True, p_weights=None, p_bias=None):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1, perturb_ray_dirs=True)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()
        style_enc.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, pred_t = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp,
                                                            edit_grid=edit_grid, scale_depth=False)
                #pred_img, pred_term, pred_w8s_density, pred_w8s_edit, depth_, min_near = self.distill_step(data,
                #                                                                                           edit_grid,
                #                                                                                           perturb_depth=False)

        if self.ema is not None:
            self.ema.restore()

        d = torch.zeros_like(preds_depth)
        d[~preds_depth.isnan()] = preds_depth[~preds_depth.isnan()]

        x_term = rays['rays_o'] + d.flatten(1, 2)[..., None] * rays['rays_d']

        indices = d.flatten().nonzero(as_tuple=True)

        #pred_w8s = pred_w8s_edit.clone()
        # difference must be larger than a threshold value \tau_{\text{weight}}
        # this eliminates the effect of floaters to a large degree
        # weights_edit_sum[torch.abs(weights_sum - weights_edit_sum) > 0.5] = 0
        #pred_w8s[torch.abs(pred_w8s_density - pred_w8s) > 0.75] = 0
        # depth must be valid for the sample to be accepted
        # bigger than the minimal near bound
        # weights_edit_sum[depth.squeeze() < nears.min()] = 0
        #pred_w8s[depth_.squeeze() < min_near] = 0
        # weights_edit_sum[weights_edit_sum > 0] = weights_sum[weights_edit_sum > 0]
        #pred_w8s[pred_w8s > 0] = pred_w8s_density[pred_w8s > 0]

        #mask = pred_w8s.nonzero(as_tuple=True)

        if show_index != -1:
            if show_weights:
                pred_colors = style_enc.get_weights(
                    x=x_term.squeeze()[indices],
                )[:, show_index][..., None].repeat((x_term.shape[0], 3))
            else:
                pred_colors = (style_enc.get_offsets(
                    x=x_term.squeeze()[indices],
                    d=rays['rays_d'].squeeze()[indices],
                )) * 0.5 + 0.5
        else:
            if use_offsets:
                pred_weights = style_enc.get_weights(
                    x=x_term.squeeze()[indices]
                )
                pred_offsets = style_enc.get_offsets(
                    x=x_term.squeeze()[indices],
                    d=rays['rays_d'].squeeze()[indices],
                )
                # apply the transformation...
                pred_weights = torch.clamp_min(p_bias[None] + p_weights[None] * pred_weights, 0)
                # normalize to [0,1] and sum to 1
                pred_weights /= pred_weights.sum(-1)[..., None]
                pred_colors = torch.clamp(pred_offsets.half() + pred_weights.half() @ style_enc.get_color_palette().half(), 0, 1)
            else:
                pred_weights = style_enc.get_weights(
                    x=x_term.squeeze()[indices]
                )
                pred_colors = pred_weights @ style_enc.get_color_palette().half()
                pred_colors = torch.clamp(pred_colors, 0, 1)
        #pred_colors *= pred_t[indices].detach()[..., None]
        pred_colors += (1 - pred_t[indices].detach()[..., None]) * bg_color.cuda()
        preds.flatten(0, 2)[indices] = pred_colors.detach().float()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
            'pred_t': pred_t
        }

        return outputs

    def val_gui_styleenc(self, pose, intrinsics, W, H, edit_grid, style_enc: StyleEncoder, bg_color=None):

        # render resolution (may need downscale to for better frame rate)
        intrinsics = intrinsics

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, H, W, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': H,
            'W': W,
        }

        self.model.eval()
        style_enc.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, pred_t = self.test_step(data, bg_color=bg_color, perturb=False,
                                                            edit_grid=edit_grid, scale_depth=False)

        if self.ema is not None:
            self.ema.restore()

        d = torch.zeros_like(preds_depth)
        d[~preds_depth.isnan()] = preds_depth[~preds_depth.isnan()]

        x_term = rays['rays_o'] + d.flatten(1, 2)[..., None] * rays['rays_d']

        indices = d.flatten().nonzero(as_tuple=True)



        weights = style_enc.get_weights(
                    x=x_term.squeeze()[indices])
        offsets = style_enc.get_offsets(
                    x=x_term.squeeze()[indices],
                    d=rays['rays_d'].squeeze()[indices])

        outputs = {
            'weights': weights,
            'offsets': offsets,
            'indices': indices,
            'alpha': pred_t
        }

        return outputs

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, edit_grid=None):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1, perturb_ray_dirs=True)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp,
                                                    edit_grid=edit_grid)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3,
                                                                                                  1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True, path=None):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if path is not None:
            file_path = f"{path}/{name}.pth"
            torch.save(state, file_path)
            return

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")