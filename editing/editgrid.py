import torch

from raymarching import raymarching
import torch as th

try:
    import _edit_grid as _backend
except ImportError:
    from .backend import _backend

import numpy as np
from collections import deque

def EDIT_GRIDSIZE() -> int:
    return 128

def EDIT_GRIDVOLUME() -> int:
    return EDIT_GRIDSIZE() ** 3

def scalbnf(x: float, n: int) -> float:
    return x * (2 ** n)

def mip_from_pos(x, max_cascade: float):
    mx = th.max(x, dim=-1).values
    exp = th.frexp(mx).exponent
    return th.minimum(th.tensor((max_cascade - 1)), th.maximum(th.tensor((0)), exp))

def grid_mip_offset(mip):
    return (EDIT_GRIDSIZE() * EDIT_GRIDSIZE() * EDIT_GRIDSIZE()) * mip

def get_bitfield_at(cell_idx, level, bitfield):
    selected_bit = cell_idx % 8
    mask = 1 << selected_bit
    return bitfield[cell_idx//8+grid_mip_offset(level)//8] & mask.byte()

def set_edit_bitfield_at(cell_idx, level, value, bitfield):
    selected_bit = cell_idx % 8
    mask = 1 << selected_bit
    bitfield[cell_idx//8+grid_mip_offset(level)//8] = (bitfield[cell_idx//8+grid_mip_offset(level)//8] & ~mask.byte()) | (value << selected_bit.byte())

def get_cell_pos(x, y, z, level):
    pos = np.array((x, y, z), dtype=float)
    pos = (pos + .5) / EDIT_GRIDSIZE() - 0.5
    pos = pos * scalbnf(1., level) + 0.5
    return pos

def get_cell_pos_(coords, level):
    pos = coords.float()
    pos = (pos + .5) / EDIT_GRIDSIZE() - 0.5
    pos = pos * scalbnf(1., level)[..., None] + 0.5
    return pos

class EditGrid:
    def __init__(self):
        self.grid = None
        self.pts = None
        self.palette = None
        self.growing_queue = deque()

    def save_grid_as_torch(self, f: str) -> None:
        th.save(self.grid.cpu(), f)

    def load_grid_as_torch(self, f: str) -> None:
        self.grid = th.load(f).cuda()

    def xor(self, negative_grid: th.cuda.ByteTensor):
        intm = th.bitwise_xor(self.grid, negative_grid)
        self.grid = th.bitwise_and(self.grid, intm)

    def and_(self, negative_grid: th.cuda.ByteTensor):
        self.grid = th.bitwise_or(self.grid, negative_grid)

    def get_grid(self):
        return self.grid

    def bw_and(self,
               other_grid: th.cuda.FloatTensor) -> None:
        th.bitwise_and(self.grid, other_grid, out=self.grid)

    def new_from_points(self,
                        pts: th.cuda.FloatTensor,
                        trainer=None,
                        bound: float = 1.):
        new_grid = th.zeros_like(trainer.model.density_bitfield)
        max_cascade = trainer.model.cascade

        level = mip_from_pos(pts, max_cascade)
        mip_bound = th.minimum(scalbnf(1, level), th.tensor((bound)))
        mip_rbound = 1. / mip_bound

        # convert to nearest grid position
        grid_pos = th.clamp(0.5 * (pts * mip_rbound[..., None] + 1) * EDIT_GRIDSIZE(), 0., EDIT_GRIDSIZE() - 1).int()

        # INVALID when grid dim is higher
        # convert the pts to the range [0,1]
        #pts = (pts + bound) / (2 * bound)
        # multiply with grid size
        #pts *= EDIT_GRIDSIZE()
        #pts = pts.byte()

        # only the ones actually in grid
        #condition = th.all(th.bitwise_and(pts >= 0, pts < EDIT_GRIDSIZE()), dim=-1).nonzero(as_tuple=True)
        #pts = pts[condition]

        # map to indices set in new grid
        indices = EDIT_GRIDVOLUME() * level + raymarching.morton3D(grid_pos).long()
        level = indices // (EDIT_GRIDVOLUME())
        pos_idx = indices % (EDIT_GRIDVOLUME())

        set_edit_bitfield_at(pos_idx, level, 1, new_grid)
        self.grid = new_grid


        for i in range(pts.shape[0]):
            pt = grid_pos[i][None, ...]
            lvl = level[i]
            # compute index
            coords_tmp = th.tensor((
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, -1),
                (0, 0, 1),
                (0, 1, 0),
                (1, 0, 0),
            )).to(pt.device)
            coords_tmp = (pt[:, None, :] + coords_tmp).flatten(0, 1)

            # TODO: boundary check

            # assert that in scene bounds....
            # must be in {0, 127} in x, y, z
            condition = th.all(th.bitwise_and(coords_tmp >= 0, coords_tmp < EDIT_GRIDSIZE()), dim=-1).nonzero(
                as_tuple=True)
            coords_tmp = coords_tmp[condition]

            for z in range(coords_tmp.shape[0]):
                self.growing_queue.append([coords_tmp[z], lvl])

    def reset(self):
        self.grid = None
        self.pts = None
        self.palette = None
        self.growing_queue = deque()

    def morphological(self):
        S = 64
        X = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Y = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Z = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = th.meshgrid(xs, ys, zs, indexing='ij')
                    coords = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                    dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    level = indices // (EDIT_GRIDVOLUME())
                    pos_idx = indices % (EDIT_GRIDVOLUME())

                    bits = get_bitfield_at(pos_idx, level, self.grid)
                    if th.count_nonzero(bits) > 0:
                        coords_hit = coords[bits.nonzero(as_tuple=True)]
                        self.add_neighbors(coords_hit, self.grid)

    def grow_region(self,
                    density_grid: th.cuda.FloatTensor,
                    density_thresh: int,
                    occ_grid: th.cuda.ByteTensor = None) -> None:

        S = 64
        X = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Y = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Z = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    # compute the indices
                    xx, yy, zz = th.meshgrid(xs, ys, zs, indexing='ij')
                    coords = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    level = indices // (EDIT_GRIDVOLUME())
                    pos_idx = indices % (EDIT_GRIDVOLUME())

                    bits = get_bitfield_at(pos_idx, level, self.growing_grid)
                    if th.count_nonzero(bits) > 0:
                        coords_hit = coords[bits.nonzero(as_tuple=True)]

                        # check whether the hit coords are higher than the density thresh
                        indices = raymarching.morton3D(coords_hit).long()  # [N]
                        pos_idx = indices % (EDIT_GRIDVOLUME())
                        level = indices // (EDIT_GRIDVOLUME())

                        condition = density_grid[0, tuple(pos_idx)] > density_thresh
                        condition2 = ~(get_bitfield_at(pos_idx, level, self.grid))
                        where_true = th.bitwise_and(condition, condition2).nonzero(as_tuple=True)
                        set_edit_bitfield_at(pos_idx[where_true], level[where_true], 1, self.grid)
                        set_edit_bitfield_at(pos_idx[where_true], level[where_true], 0, self.growing_grid)

                        self.add_neighbors(coords_hit[where_true], self.growing_grid)

    def add_neighbors(self,
                      coords_hit,
                      grid):
        #offsets_ = th.tensor((-1, 0, 1)).cuda()
        #xx, yy, zz = th.meshgrid(offsets_, offsets_, offsets_, indexing='ij')
        #coords_tmp = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
        #                    dim=-1)
        coords_tmp = th.tensor((
            (-1, 0, 0),
            (0, -1, 0),
            (0,  0, -1),
            (0,  0, 1),
            (0,  1, 0),
            (1,  0, 0),
        )).to(coords_hit.device)
        coords_tmp = (coords_hit[:, None, :] + coords_tmp).flatten(0, 1)

        # assert that in scene bounds....
        # must be in {0, 127} in x, y, z
        condition = th.all(th.bitwise_and(coords_tmp >= 0, coords_tmp < EDIT_GRIDSIZE()), dim=-1).nonzero(as_tuple=True)
        coords_tmp = coords_tmp[condition]

        indices = raymarching.morton3D(coords_tmp).long()  # [N]
        pos_idx = indices % (EDIT_GRIDVOLUME())
        level = indices // (EDIT_GRIDVOLUME())

        set_edit_bitfield_at(pos_idx, level, 1, grid)

    def grid_from_growing_queue(self,
                                density_thresh: float,
                                grid,
                                density_grid: th.cuda.FloatTensor):
        self.grid = torch.zeros_like(grid.get_grid())
        growing_queue = grid.growing_queue

        for i in range(len(growing_queue)):
            level = torch.zeros((1), dtype=torch.int32).cuda()
            idx, level[0] = growing_queue[i]
            indices = raymarching.morton3D(idx[None, ...]).long()

            pos_idx = indices % (EDIT_GRIDVOLUME())

            density = density_grid[level, pos_idx]
            condition = (~get_bitfield_at(pos_idx, level, self.grid).bool()).item()

            # Sample accepted only if at requested level, satisfying density threshold and not already selected!
            if (density.squeeze() >= density_thresh).item() and condition:
                set_edit_bitfield_at(pos_idx, level, 1, self.grid)
                coords_tmp = th.tensor((
                    (-1, 0, 0),
                    (0, -1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                    (0, 1, 0),
                    (1, 0, 0),
                )).to(idx.device)
                # coords_tmp = (idx[None, None, ...] + coords_tmp).flatten(0, 1)
                coords_tmp = (idx[None, None, :] + coords_tmp[None, ...]).flatten(0, 1)
                levels = th.zeros(6).cuda() + level[0]

                # assert that in scene bounds....
                # must be in {0, 127} in x, y, z
                condition = th.all(th.bitwise_and(coords_tmp >= 0, coords_tmp < EDIT_GRIDSIZE()), dim=-1).nonzero(
                    as_tuple=True)
                coords_tmp = coords_tmp[condition]

                for z in range(coords_tmp.shape[0]):
                    self.growing_queue.append([coords_tmp[z], levels[z]])


    def grow_region_queue(self,
                          density_grid: th.cuda.FloatTensor,
                          density_thresh: int,
                          occ_grid: th.cuda.ByteTensor = None,
                          grow_iterations=5000) -> None:
        if len(self.growing_queue) == 0:
            # deque is empty, push the ones from the density grid to the deque
            print('Growing Queue is for some reason empty')

        max_N = 32
        ctr_ = 0

        while ctr_ < grow_iterations:
            if len(self.growing_queue) == 0:
                break

            num_indices = min(min(max_N, len(self.growing_queue)), grow_iterations - ctr_)

            idx = torch.zeros((num_indices, 3), dtype=torch.int32).cuda()
            lvl = torch.zeros((num_indices), dtype=torch.int32).cuda()
            for i in range(num_indices):
                idx[i], lvl[i] = self.growing_queue.popleft()

            # idx = self.growing_queue.popleft()
            # indices = raymarching.morton3D(idx[None, ...]).long()
            indices = raymarching.morton3D(idx).long()

            pos_idx = indices % (EDIT_GRIDVOLUME())

            density = density_grid[lvl, pos_idx]
            #condition = (~get_bitfield_at(pos_idx, level, self.grid).bool()).item()
            condition = (~get_bitfield_at(pos_idx, lvl, self.grid).bool())

            # Sample accepted only if at requested level, satisfying density threshold and not already selected!
            cond = th.bitwise_and(density.squeeze() >= density_thresh, condition)
            if cond.sum() > 0:
                idx = idx[cond]
                pos_idx = pos_idx[cond]
                level = lvl[cond]
                set_edit_bitfield_at(pos_idx, level, 1, self.grid)
            #if (density.squeeze() >= density_thresh).item() and condition:
                #set_edit_bitfield_at(pos_idx, level, 1, self.grid)

                # add neighbors to the queue
                coords_tmp = th.tensor((
                    (-1, 0, 0),
                    (0, -1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                    (0, 1, 0),
                    (1, 0, 0),
                )).to(idx.device)
                #coords_tmp = (idx[None, None, ...] + coords_tmp).flatten(0, 1)
                coords_tmp = (idx[:, None, :] + coords_tmp[None, ...]).flatten(0, 1)
                levels = th.zeros(cond.sum() * 6).cuda() + lvl[0]

                # assert that in scene bounds....
                # must be in {0, 127} in x, y, z
                condition = th.all(th.bitwise_and(coords_tmp >= 0, coords_tmp < EDIT_GRIDSIZE()), dim=-1).nonzero(
                    as_tuple=True)
                coords_tmp = coords_tmp[condition]

                for z in range(coords_tmp.shape[0]):
                    self.growing_queue.append([coords_tmp[z], levels[z]])

            ctr_ += num_indices
        print(f'finished growing for {grow_iterations} steps!')


    def get_selection_points(self):
        if self.pts is not None:
            return self.pts
        sel_points = []

        S = 32
        X = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Y = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
        Z = th.arange(EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = th.meshgrid(xs, ys, zs, indexing='ij')
                    coords = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    level = indices // (EDIT_GRIDVOLUME())
                    pos_idx = indices % (EDIT_GRIDVOLUME())

                    bits = get_bitfield_at(pos_idx.cpu().numpy(), level.cpu().numpy(), self.grid)
                    if np.count_nonzero(bits) > 0:
                        indices = raymarching.morton3D_invert(pos_idx[bits.nonzero()[0]])
                        levels = level[bits.nonzero()[0]]
                        sel_points.append(get_cell_pos_(indices, levels))
        return np.concatenate([s.cpu().numpy() for s in sel_points])




