# coding:utf-8
# @Time    : 2025/10/4 15:12
# @Author  : XiaoYuan
# @FileName: reconstruct.py
# @description: Initial reconstruction of photoacoustic sinogram using the Filtered Back-Projection (FBP) algorithm.

import os
import torch
import logging
import argparse
import numpy as np
from kwave import kWaveGrid
from scipy.io import savemat
from matplotlib import pyplot as plt
from recon.fbp import FilteredBackProjection
from utils.dataio import read_yaml_param, read_matlab_hdf5_with_refs


def fbp(sinogram: np.ndarray, v0: float, params: dict, results_dir: str, serial_number:float) -> None:
    """
        Run Filtered Back-Projection (FBP) reconstruction with given sound speed.
    Args:
        sinogram: photoacoustic sinogram.
        v0: Assumed constant sound speed.
        params: reconstruction parameters.
        results_dir: Directory to save reconstruction results.
        serial_number: Serial number.

    Returns:
        None
    """

    logger = logging.getLogger('FBP')
    logger.info(" Reconstructing with Filtered Back-Projection (%s).",  f'v_fbp={v0:.1f}m·s⁻¹')

    # Preparations physic_operate params.
    scale = 0.72  # Scaling ratio of single pixel physical dimension
    Nx, Ny = params['Nx'], params['Ny']
    dx, dy = params['dx'] * scale, params['dy'] * scale
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])
    x_vec, y_vec = kgrid.x_vec, kgrid.y_vec
    R_ring = params['R_ring']
    start_angle_deg, end_angle_deg = params['angle_range']

    # Convert angle to radians
    start_angle_rad = np.deg2rad(start_angle_deg)
    end_angle_rad = np.deg2rad(end_angle_deg)

    # define model
    fbp = FilteredBackProjection(R_ring=R_ring, N_transducer=params['N_transducer'], T_sample=params['sampling_rate'],
                                 angle_range=(torch.tensor(start_angle_rad), torch.tensor(end_angle_rad)), x_vec=x_vec,
                                 y_vec=y_vec, theta_limit_deg=params['theta_limit_deg']).cuda()
    fbp.eval()

    sinogram = sinogram[::, params['t0']:].copy()

    with torch.no_grad():
        ip_rec = fbp(sinogram=sinogram, v0=v0).detach().cpu().numpy()

    # Saved directory.
    share_path = os.path.join(results_dir, f'common/{serial_number: 04d}')
    result_mat_path = os.path.join(results_dir, 'mats')
    result_image_path = os.path.join(results_dir, 'images')

    #  Check if the saved directory exists.
    if not os.path.exists(share_path):
        os.makedirs(share_path, exist_ok=True)
    if not os.path.exists(result_mat_path):
        os.makedirs(result_mat_path, exist_ok=True)
    if not os.path.exists(result_image_path):
        os.makedirs(result_image_path, exist_ok=True)

    savemat(os.path.join(share_path, 'rec_image.mat'), {'recon': ip_rec})
    savemat(os.path.join(result_mat_path, f'{serial_number + 1: 04d}.mat'), {'recon': ip_rec})

    # Normalization
    ip_rec = (ip_rec - ip_rec.mean()) / ip_rec.std()
    plt.imsave(os.path.join(share_path, 'rec_image.jpg'), ip_rec, cmap='gray', vmin=-6, vmax=7)
    plt.imsave(os.path.join(result_image_path, f'{serial_number + 1:04d}.jpg'), ip_rec, cmap='gray', vmin=-6, vmax=7)


def multi_temporal_sequence_reconstruction(yml_path: str, singoram_path:str, result_path:str):
    """
        Multiple groups of time-series reconstructed tomographic cross-sections
    :return: None
    """

    # coad signals and configuration files.
    ymls = read_yaml_param(yml_path)['recon_params']
    signal_data = read_matlab_hdf5_with_refs(singoram_path)

    # check if the save folder exists.
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    for index, sinogram in enumerate(signal_data):
        logging.info(f'Processing the {index + 1:05d} set of photoacoustic signals')
        # signal Preprocessing (Rotation).
        sinogram = sinogram[::-1, ...]
        n_channels = sinogram.shape[0]
        shift = n_channels // 4
        sinogram = np.roll(sinogram, shift=-shift, axis=0)

        fbp(sinogram, ymls['v0'], ymls, result_path, index)

    logging.info(f'All reconstructions completed save path {result_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="filter back-projection reconstruction")

    parser.add_argument('--config', default='./config/recon_parameter.yml')
    parser.add_argument('--singoram', default='./data/05Hz_03Hz_raw_data_blocked_1_to_120.mat')
    parser.add_argument('-o', '--output', default='./result_new/05Hz_03Hz_blocked_1_to_120/fbp')

    args = parser.parse_args()

    multi_temporal_sequence_reconstruction(args.config, args.singoram, args.output)  # Multi view reconstruction
