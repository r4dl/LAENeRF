import argparse

import cv2
import json
import os
import icecream
import numpy as np

datatype = 'llff'
def evaluate(directory: str, scene: str, save_dir: str = None, base_images: str = None):
    # open the transform.json such that we can get the names of the test images
    with open(f'data/{datatype}/{scene}/transforms_test.json') as fp:
        file = json.load(fp)

    # get the reference/base images
    # if the directory does not exist, take the reference images
    if base_images is None:
        paths = file['frames']
        imgs = [f'data/{datatype}/{scene}/{i["file_path"]}' for i in paths]
    else:
        imgs = [os.path.join(base_images, i) for i in sorted(os.listdir(base_images))]

    # get the masks
    masks = [os.path.join('scripts', 'eval', 'masks', f'{datatype}', f'{scene}', i["file_path"][9:]) for i in file['frames']]

    # get the recolored images
    imgs_recolored = [os.path.join(directory, i) for i in sorted(os.listdir(directory))]

    # create directory to save diff images to
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    errors = np.zeros((len(imgs)))

    for img_index, img in enumerate(imgs):
        # load recolored image
        out_senc = cv2.imread(imgs_recolored[img_index])
        out_senc = cv2.cvtColor(out_senc, cv2.COLOR_BGR2RGB) / 255.

        # load reference image
        ref = cv2.imread(img)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB) / 255.

        if out_senc.shape != ref.shape:
            out_senc = cv2.resize(out_senc, (ref.shape[1], ref.shape[0]))

        # load the masks
        # masks from ICE-NeRF are saved in the G channel
        mask = cv2.imread(masks[img_index])
        if mask.shape != ref.shape:
            mask = cv2.resize(mask, (ref.shape[1], ref.shape[0]))
            mask = mask.max(-1)[..., None] / mask.max(-1)[..., None].max()
            mask = 1 - mask

        error_img = (np.square(out_senc - ref) * mask)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, f'error_{img_index:03d}.png'),
                        (error_img * 255).astype(np.byte))
        errors[img_index] = (error_img.sum() / mask.sum() / 3)

    # save the results if a save dir is there
    if save_dir is not None:
        results = {
            'errors': errors.tolist(),
            'mean': errors.mean()
        }
        with open(os.path.join(save_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp, indent=2)

    # in any case, print results
    icecream.ic(np.mean(errors))
    icecream.ic(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='scene name')
    parser.add_argument('--results_dir', type=str, help='directory in which the results reside',
                        required=True)
    parser.add_argument('--comparison_dir', type=str,
                        help='directory for images before recoloring (optional)', required=False)
    parser.add_argument('--save_dir', type=str, help='where to save the results (optional)', required=False)

    options = parser.parse_args()

    evaluate(directory=options.results_dir,
             scene=options.scene,
             save_dir=options.save_dir,
             base_images=options.comparison_dir)



