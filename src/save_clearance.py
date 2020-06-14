''' Python script to save clearance scores for low-res data'''

import os
import numpy as np
import glob
import skimage.io as io
import argparse

from tqdm import tqdm
from utils import get_imageset_directories


def save_clearance_scores(dataset_directories):
    '''
    Saves low-resolution clearance scores as .npy under imageset dir
    Args:
        dataset_directories: list of imageset directories
    '''

    for imset_dir in tqdm(dataset_directories):

        idx_names = np.array([os.path.basename(path)[2:-4] for path in glob.glob(os.path.join(imset_dir, 'QM*.png'))])
        idx_names = np.sort(idx_names)
        lr_maps = np.array([io.imread(os.path.join(imset_dir, f'QM{i}.png')) for i in idx_names], dtype=np.uint16)

        scores = lr_maps.sum(axis=(1, 2))
        np.save(os.path.join(imset_dir, "clearance.npy"), scores)


def main():
    '''
    Calls save_clearance on train and test set.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="root dir of the dataset", default='data/')
    args = parser.parse_args()

    data_dir = args.data_dir
    assert os.path.isdir(data_dir)
    if os.path.exists(os.path.join(data_dir, "train")):
        train_set_directories = get_imageset_directories(os.path.join(data_dir, "train"))
        save_clearance_scores(train_set_directories) # train data

    if os.path.exists(os.path.join(data_dir, "test")):
        test_set_directories = get_imageset_directories(os.path.join(data_dir, "test"))
        save_clearance_scores(test_set_directories) # test data


if __name__ == '__main__':
    main()
