import torch as th
from raymarching import raymarching
import matplotlib.pyplot as plt
import editing.editgrid as egrid
import numpy as np

import matplotlib as mpl
mpl.use('Qt5Agg')

def plot_grid(str_of_density_grid: str, str_of_edit_grid: str):
    torch_grid_dens = th.load(str_of_density_grid).cuda()
    torch_grid_edit = th.load(str_of_edit_grid).cuda()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # combine the objects into a single boolean array
    voxelarray = np.zeros((egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE()), dtype=bool)
    editarr = np.zeros((egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE()), dtype=bool)
    densarr = np.zeros((egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE(),
                           egrid.EDIT_GRIDSIZE()), dtype=bool)
    colors = np.empty(voxelarray.shape, dtype=object)

    S = 32
    X = th.arange(egrid.EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
    Y = th.arange(egrid.EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)
    Z = th.arange(egrid.EDIT_GRIDSIZE(), dtype=th.int32, device='cuda').split(S)

    for xs in X:
        for ys in Y:
            for zs in Z:
                xx, yy, zz = th.meshgrid(xs, ys, zs, indexing='ij')
                coords = th.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                dim=-1)  # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long()  # [N]
                level = indices // (egrid.EDIT_GRIDVOLUME())
                pos_idx = indices % (egrid.EDIT_GRIDVOLUME())

                bits = egrid.get_bitfield_at(pos_idx, level, torch_grid_dens).reshape(32, 32, 32)
                bits_edit = egrid.get_bitfield_at(pos_idx, level, torch_grid_edit).reshape(32, 32, 32)
                editarr[xs[0].item():xs[0].item()+S,
                ys[0].item():ys[0].item()+S,
                zs[0].item():zs[0].item()+S] = bits_edit.cpu().numpy()
                densarr[xs[0].item():xs[0].item()+S,
                ys[0].item():ys[0].item()+S,
                zs[0].item():zs[0].item()+S] = bits.cpu().numpy()

    voxelarray = densarr | editarr

    bitwise_and_grid = np.zeros_like(voxelarray)
    bitwise_and_grid[editarr.nonzero()[0].min():editarr.nonzero()[0].max(), editarr.nonzero()[1].min():editarr.nonzero()[1].max(), editarr.nonzero()[2].min():editarr.nonzero()[2].max()] = True
    voxelarray = voxelarray & bitwise_and_grid

    colors[...] = 'green'
    colors[densarr] = 'blue'
    colors[editarr] = 'red'

    # voxelgrid should be fin
    ax.voxels(voxelarray.transpose(0, 2, 1),
              facecolors=colors.transpose(0, 2, 1),
              edgecolors=colors.transpose(0, 2, 1))
    #plt.show()
    plt.savefig(f"{str_of_density_grid[:str_of_density_grid.find('.')]}{str_of_edit_grid[:str_of_edit_grid.find('.')]}.png", bbox_inches='tight')

if __name__ == '__main__':
    str_grid = 'density_bitfield.pth'
    str_grid_edit = 'grid_queue.pth'
    plot_grid(str_grid, str_grid_edit)