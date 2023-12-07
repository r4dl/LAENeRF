

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

import matplotlib as mpl
#mpl.use('Qt5Agg')

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

def cube_figs() -> None:

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    pts = np.load('xterm.npy').T
    pts_col = np.load('xtarget.npz.npy')

    # Hide grid lines
    n=100

    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        #ax.scatter(xs, ys, zs, marker=m)

    ax.scatter3D(pts[:, 0], pts[:, 2],1-pts[:, 1], c=pts_col, s=0.5)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax._axis3don = False

    plt.show(bbox_inches='tight', pad_inches=0, dpi=800)
    #plt.savefig("test.png", bbox_inches='tight',  dpi=800)

if __name__ == '__main__':
    cube_figs()