""" Utility functions """

import csv
import torch
import os
import warnings
import numpy as np
from skimage import io, img_as_uint
from tqdm import tqdm
from zipfile import ZipFile
from Evaluator import shift_cPSNR
from DataLoader import ImageSet


def read_baseline_CPSNR(path):
    """
    Reads the baseline cPSNR scores from `path`.
    Args:
        filePath: str, path/filename of the baseline cPSNR scores
    Returns:
        scores: dict, of {'imagexxx' (str): score (float)}
    """
    scores = dict()
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            scores[row[0].strip()] = float(row[1].strip())
    return scores


def get_imageset_directories(data_dir):
    """
    Returns a list of paths to directories, one for every imageset in `data_dir`.
    Args:
        data_dir: str, path/dir of the dataset
    Returns:
        imageset_dirs: list of str, imageset directories
    """
    
    imageset_dirs = []
    for channel_dir in ['RED', 'NIR']:
        path = os.path.join(data_dir, channel_dir)
        for imageset_name in os.listdir(path):
            imageset_dirs.append(os.path.join(path, imageset_name))
    return imageset_dirs


class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, min_L=32):
        """
        Args:
            min_L: int, pad length
        """
        
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """
        
        lr_batch = []  # batch of low-resolution views
        alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names

        train_batch = True

        for imageset in batch:

            lrs = imageset['lr']
            L, H, W = lrs.shape

            if L >= self.min_L:  # pad input to top_k
                lr_batch.append(lrs[:self.min_L])
                alpha_batch.append(torch.ones(self.min_L))
            else:
                pad = torch.zeros(self.min_L - L, H, W)
                lr_batch.append(torch.cat([lrs, pad], dim=0))
                alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

            hr = imageset['hr']
            if train_batch and hr is not None:
                hr_batch.append(hr)
            else:
                train_batch = False

            hm_batch.append(imageset['hr_map'])
            isn_batch.append(imageset['name'])

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        alpha_batch = torch.stack(alpha_batch, dim=0)

        if train_batch:
            hr_batch = torch.stack(hr_batch, dim=0)
            hm_batch = torch.stack(hm_batch, dim=0)

        return padded_lr_batch, alpha_batch, hr_batch, hm_batch, isn_batch


def get_sr_and_score(imset, model, min_L=16):
    '''
    Super resolves an imset with a given model.
    Args:
        imset: imageset
        model: HRNet, pytorch model
        min_L: int, pad length
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
        scPSNR: float, shift cPSNR score
    '''

    if imset.__class__ is ImageSet:
        collator = collateFunction(min_L=min_L)
        lrs, alphas, hrs, hr_maps, names = collator([imset])
    elif isinstance(imset, tuple):  # imset is a tuple of batches
        lrs, alphas, hrs, hr_maps, names = imset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)

    sr = model(lrs, alphas)[:, 0]
    sr = sr.detach().cpu().numpy()[0]

    if len(hrs) > 0:
        scPSNR = shift_cPSNR(sr=np.clip(sr, 0, 1),
                             hr=hrs.numpy()[0],
                             hr_map=hr_maps.numpy()[0])
    else:
        scPSNR = None

    return sr, scPSNR


def generate_submission_file(model, imset_dataset, out='../submission'):
    '''
    USAGE: generate_submission_file [path to testfolder] [name of the submission folder]
    EXAMPLE: generate_submission_file data submission
    '''

    print('generating solutions: ', end='', flush='True')
    os.makedirs(out, exist_ok=True)

    for imset in tqdm(imset_dataset):
        folder = imset['name']
        sr, _ = get_sr_and_score(imset, model)
        sr = img_as_uint(sr)

        # normalize and safe resulting image in temporary folder (complains on low contrast if not suppressed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join(out, folder + '.png'), sr)
            print('*', end='', flush='True')

    print('\narchiving: ')
    sub_archive = out + '/submission.zip'  # name of submission archive
    zf = ZipFile(sub_archive, mode='w')
    try:
        for img in os.listdir(out):
            if not img.startswith('imgset'):  # ignore the .zip-file itself
                continue
            zf.write(os.path.join(out, img), arcname=img)
            print('*', end='', flush='True')
    finally:
        zf.close()
    print('\ndone. The submission-file is found at {}. Bye!'.format(sub_archive))

