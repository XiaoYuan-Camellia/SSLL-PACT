# coding:utf-8
# @Time    : 2025/9/24 16:10
# @Author  : XiaoYuan
# @FileName: dataio.py
# @description: Data stream processing file

import os
import h5py
import yaml
import torch
import logging
import hdf5storage
import numpy as np
from PIL import Image


def read_yaml_param(file_path):
    """
        Basic function for reading YAML files
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"File does not exist: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        return None
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None


def read_matlab_hdf5_with_refs(file_path):
    """
        load mat files in H5 format
    Args:
        file_path: mat file path

    Returns:
        all_data: multidimensional photoacoustic signals
    """

    with h5py.File(file_path, 'r') as f:
        # print("Keys in the file:", list(f.keys()))

        if 'processed_data_all' in f:
            refs_array = f['processed_data_all'][()]
            print(f"Shape of processed_data_all: {refs_array.shape}")

            # Iterate through all references and parse the actual data.
            all_data = []
            for i, ref in enumerate(refs_array.flat):
                try:
                    # Resolve the actual object from the HDF5 object reference
                    # target_obj = f.get(ref)
                    # obj_key = target_obj.name
                    # print(f"Reference {i} -> H5 path key: {obj_key}")
                    target_obj = f[ref]
                    if isinstance(target_obj, h5py.Dataset):
                        data = target_obj[()]
                        all_data.append(data)
                except Exception as e:
                    print(f"Failed to parse reference {i}: exception {e}")

            return all_data

        return None


def load_images_simple(folder_path):
    """
        Directly read all images in the folder, convert them to single-channel images, normalize pixel values to the
        range [0, 1], and return a Tensor array.
    Args:
        folder_path: image folder path

    Returns:
        A Tensor with the shape of (n, 1, H, W), where n denotes the number of images.
    """

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    image_tensors = []

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        image_tensors.append(img_tensor)
    return torch.stack(image_tensors, dim=0)


def save_mat(file:str, data:np.ndarray, key:str='data') -> None:
    """Save data to `.mat` file.

    Args:
        file (str): Path to file.
        data (np.ndarray): The dictionary of data to be saved.
        key (str, optional): The key to be used in the dictionary. Defaults to `'data'`.
    """
    logger = logging.getLogger('DataIO')
    if os.path.exists(file):
        os.remove(file)
    hdf5storage.savemat(file, {key: data})
    logger.debug(' Successfully saved data to "%s".', file)
