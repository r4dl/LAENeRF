import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2, json
import lpips

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

def compute_flow_gradients(flow):
    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv

@torch.no_grad()
def detect_occlusion(fw_flow, bw_flow):
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    fw_flow_t = torch.from_numpy(fw_flow).cuda()
    bw_flow_t = torch.from_numpy(bw_flow).cuda()

    ## warp fw-flow to img2
    flow = torch.from_numpy(fw_flow).permute(2, 0, 1).float()
    flow[0, :, :] += torch.arange(flow.shape[2])
    flow[1, :, :] += torch.arange(flow.shape[1])[:, None]
    fw_flow_w = cv2.remap((bw_flow_t).cpu().numpy(), flow.permute(1, 2, 0).cpu().numpy(), None, cv2.INTER_LINEAR)

    ## convert to numpy array
    fw_flow_w = (fw_flow_w)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion

@torch.no_grad()
def validate_longrange(model, frames_directory, frames_directory_gt, step=7):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    iters=32

    lpipsmeter = lpips.LPIPS(net='alex').cuda()

    frames_gt = sorted([i for i in os.listdir(frames_directory_gt) if 'png' in i])
    frames_ours = sorted([i for i in os.listdir(frames_directory) if 'png' in i])
    flow_prev, sequence_prev = None, None

    errors_ours = []
    lpips_ours = []

    for test_id in range(len(frames_directory) - step):
        image1 = frame_utils.read_gen(os.path.join(frames_directory_gt, frames_gt[test_id]))
        image2 = frame_utils.read_gen(os.path.join(frames_directory_gt, frames_gt[test_id + step]))
        image1 = np.array(image1).astype(np.uint8)[..., :3]
        image2 = np.array(image2).astype(np.uint8)[..., :3]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_low, flow_pr = model(image2, image1, iters=iters, flow_init=flow_prev, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
        flow_bw = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        flow_clone = np.copy(flow)
        occ = detect_occlusion(fw_flow=flow_clone, bw_flow=flow_bw)

        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow[0, :, :] += torch.arange(flow.shape[2])
        flow[1, :, :] += torch.arange(flow.shape[1])[:, None]

        image1 = frame_utils.read_gen(os.path.join(frames_directory, frames_ours[test_id]))
        image2 = frame_utils.read_gen(os.path.join(frames_directory, frames_ours[test_id + step]))
        image1 = np.array(image1).astype(np.uint8)[..., :3]
        image2 = np.array(image2).astype(np.uint8)[..., :3]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float() / 255.
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float() / 255.

        warped_result = cv2.remap((image1).permute(1,2,0).cpu().numpy(), flow.permute(1,2,0).cpu().numpy(), None, cv2.INTER_LINEAR)
        lpi = lpipsmeter(torch.from_numpy(warped_result).permute(-1,0,1).cuda(), image2.cuda()).item()

        mse = (((torch.from_numpy(warped_result).permute(2,0,1) - image2)**2).sum(0) * occ)
        error = (mse.sum() / occ.sum())

        errors_ours.append(error.item())
        lpips_ours.append(lpi)

    print(f'RSME of {np.mean(np.array(errors_ours)):.4f}')
    print(f'LPIPS of {np.mean(np.array(lpips_ours))}')

    dict_ = {
        'rsme': np.mean(np.array(errors_ours)),
        'lpips': np.mean(np.array(lpips_ours))
    }

    with open('results.json', 'w') as outfile:
        json.dump(dict_, outfile, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--step', type=int, help='number of steps to compute the optical flow')
    parser.add_argument('--directory', type=str, help='directory in which the images to be evaluated lie')
    parser.add_argument('--directory_gt', type=str, help='directory in which the gt images lie')
    args = parser.parse_args()

    # custom args
    args.small = False
    args.mixed_precision = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    with torch.no_grad():
        validate_longrange(model.module, frames_directory=args.directory, frames_directory_gt=args.directory_gt,
                           step=args.step)