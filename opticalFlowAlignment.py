# coding:utf-8
# @Time    : 2025/9/14 12:54
# @Author  : XiaoYuan
# @FileName: opticalFlowAlignment.py
# @description: Compute optical flow from neighboring frames to the reference frame and perform alignment accordingly.

import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt


class OpticalFlowAlignment:
    def __init__(self):
        self.current_config = {
            "pyr_scale": 0.16,
            "levels": 3,
            "winsize": 43,
            "iterations": 15,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        }

    def flow_fb(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
            compute forward optical flow
        Args:
            a: reference frame image
            b: neighboring frame image

        Returns:
            optical flow field
        """
        return cv2.calcOpticalFlowFarneback(a, b, None, **self.current_config)

    def warp_image_with_flow(self, target_img, flow):
        """
             Warp neighboring images to the coordinate system of the reference image using the optical flow field.
        Args:
            target_img: neighboring frame image
            flow: optical flow field

        Returns:
            warped_img: warped aligned image
        """
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # using the optical flow field
        map_x += flow[:, :, 0]
        map_y += flow[:, :, 1]

        # Perform warping via remap
        warped_img = cv2.remap(target_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return warped_img


def process_image_sequence(input_folder, output_folder):
    """
        Perform processing on the image sequence
    Args:
        input_folder: Input image folder path
        output_folder: Output image folder path

    Returns:
        None
    """

    os.makedirs(output_folder, exist_ok=True)
    warped_img_handle_folder = output_folder

    # Obtain all image files within the directory
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()  # in-place operation.

    clacflow = OpticalFlowAlignment()
    vmin, vmax = -6, 7
    fusion_num_img = 2

    # Use every frame as the reference frame one by one,
    # and calculate the aligned results of neighboring frames after registration.
    for i in range(len(image_files)):

        # reference image
        img1_path = os.path.join(input_folder, image_files[i])
        ref_image = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

        # Create directory to store aligned images.
        current_warp_image_folder = os.path.join(warped_img_handle_folder, f'{i+1:04d}/images')
        os.makedirs(current_warp_image_folder, exist_ok=True)

        # Save reference frames
        First_image = (ref_image - ref_image.mean()) / ref_image.std()
        plt.imsave(os.path.join(current_warp_image_folder, f"{i+1:04}.jpg"), First_image, vmin=vmin, vmax=vmax, cmap='gray')  # 保存参考图像

        for j in range(1, fusion_num_img+1):
            tar_index = i + j
            if 0 < tar_index < len(image_files):
                tar_image = cv2.imread(os.path.join(input_folder, image_files[tar_index]), cv2.IMREAD_GRAYSCALE)

                # Compute forward optical flow
                flow = clacflow.flow_fb(ref_image, tar_image)

                # align images
                warped_img = clacflow.warp_image_with_flow(tar_image, flow)

                # Standardize and save the aligned images
                warped_img = (warped_img - warped_img.mean()) / warped_img.std()
                plt.imsave(os.path.join(current_warp_image_folder, f"{tar_index+1:04}.jpg"), warped_img, vmin=vmin, vmax=vmax, cmap='gray')

        print(f'Processing the {i + 1:04d} reference image !')

    print("All image sequence processing is completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="optical flow align images")

    parser.add_argument('--input_folder', default='./result_new/05Hz_03Hz_L_30_S_25_blocked_sim_1_to_120/fbp/images')
    parser.add_argument('--output_folder', default='./result_new/05Hz_03Hz_L_30_S_25_blocked_sim_1_to_120/align_img')

    args = parser.parse_args()

    process_image_sequence(args.input_folder, args.output_folder)
