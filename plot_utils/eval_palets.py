import os
import torch

def compare_palettes():

    path = os.path.join('lego_palet')

    palets = []

    for i in sorted(os.listdir(path)):
        path_ = os.path.join(path, i)

        palet = torch.load(os.path.join(path_, 'style_enc.pth')).get_color_palette()
        palets.append(palet)

    for i in range(len(palets)):
        min_dist = 1.
        max_dist = -1.

        palet_1 = palets[i]

        for j in range(len(palets)):
            if i == j:
                continue
            palet_2 = palets[j]
            dists = torch.linalg.norm(palet_1 - palet_2[:, None, :], axis=-1)
            dists_intra = torch.linalg.norm(palet_1 - palet_1[:, None, :], axis=-1)

            # eliminate the diagonal
            dists_intra += torch.eye((dists_intra.diag().shape[0])).cuda() * 15

            min_dist = min(min_dist, dists.min().item())
            max_dist = max(max_dist, dists_intra.min().item())
        print(f"min {i}:\t {min_dist}")
        print(f"intra-dist {i}:\t {max_dist}")

if __name__ == '__main__':
    compare_palettes()