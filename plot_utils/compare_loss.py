import numpy as np
import matplotlib.pyplot as plt
import os

def compare_loss(
        arr1: str,
        arr2: str
):
    v1 = np.load(arr1)
    v2 = np.load(arr2)

    assert v1.shape[0] == v2.shape[0]

    fig, axis = plt.subplots(1, 1)
    axis.plot(np.arange(0, v1.shape[0]) * 25, np.array(v1), 'r', label='not learned')
    axis.plot(np.arange(0, v1.shape[0]) * 25, np.array(v2), 'b', label='learned')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss/Epochs with/without palette learning')
    plt.legend()
    plt.savefig(os.path.join("../loss_comp.png"), bbox_inches='tight', pad_inches=0.)
    plt.close()

if __name__ == '__main__':
    compare_loss(
        arr1='ablation/conf1/losses.npz.npy',
        arr2='ablation/conf3-learnpalette/losses.npz.npy'
    )