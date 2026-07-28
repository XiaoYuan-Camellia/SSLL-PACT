# coding:utf-8
# @Time    : 2026/1/6 13:34
# @Author  : XiaoYuan
# @FileName: train.py
# @description: Self-Supervised Latent Space Learning Framework (SSLL) for Photoacoustic Tomography


import os
import torch
import logging
import argparse
import warnings
import matplotlib.pyplot as plt
from utils.dataio import load_images_simple, save_mat
from models.ssll import SelfSupervisedLatentSpaceLearn

warnings.filterwarnings("ignore", category=UserWarning)


def train(args):
    logging.basicConfig(level=logging.INFO)
    root_path = args.root_path

    # Acquire the input data folder for subsequent processing
    file_list = os.listdir(root_path)

    for file_item in file_list:
        print(f'Processing: {file_item}')
        common_path = os.path.join(root_path, file_item)
        path = os.path.join(common_path, 'images')
        data_set = load_images_simple(path).cuda()

        # network parameters
        num_eopchs = args.num_eopchs  # Total number of iterations
        beta = args.beta
        lr = args.lr

        # Output directory created.
        result_sharp_path = os.path.join(common_path, f'sharp_{beta:.2f}')
        result_h5_path = os.path.join(common_path, 'result_h5')
        os.makedirs(result_sharp_path, exist_ok=True)
        os.makedirs(result_h5_path, exist_ok=True)
        vmin, vmax = [-6, 7]

        # define model
        ssll = SelfSupervisedLatentSpaceLearn(image_batch=data_set.shape[0], lr=lr).cuda()

        ssll.train()

        # record loss
        total_losses = []
        kl_losses = []
        rec_losses = []

        for epoch in range(1, num_eopchs + 1):
            fused_clear_img, rec_loss, kl_loss, total_loss = ssll(data_set, beta)
            total_losses.append(total_loss.item())
            kl_losses.append(kl_loss.item())
            rec_losses.append(rec_loss.item())

            with torch.no_grad():
                if epoch % 100 == 0:   # Save images every 100 iterations.
                    grid_clear = fused_clear_img.detach().squeeze(0).squeeze(0).cpu().numpy()

                    grid_clear = (grid_clear - grid_clear.mean()) / grid_clear.std()
                    plt.imsave(os.path.join(result_sharp_path, f'fused_clear_{epoch:05d}.jpg'), grid_clear, cmap='gray',
                               vmin=vmin, vmax=vmax)

            logging.info(f"epoch:{epoch:03d} | L_total:{total_loss:.6f} | L_re:{rec_loss:.6f} | L_kl:{kl_loss:.6f} ")

            torch.cuda.empty_cache()

        ssll.eval()
        with torch.no_grad():
            fused_clear_img = ssll.predict(data_set)
            fused_clear_img = fused_clear_img.detach().cpu().numpy()
            # save ip_rec
            save_mat(os.path.join(result_h5_path, 'result.mat'), fused_clear_img, 'IP')
            fused_clear_img = (fused_clear_img - fused_clear_img.mean()) / fused_clear_img.std()
            plt.imsave(os.path.join(result_h5_path, 'result.jpg'), fused_clear_img, cmap='gray', vmin=vmin, vmax=vmax)
            plt.close()

        # clear
        del data_set, ssll, fused_clear_img
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-Supervised Latent Space Learning Framework (SSLL)")

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_eopchs', type=int, default=6000, help='number of eopchs')
    parser.add_argument('--beta', type=float, default=0.7, help='beta controller parameter')
    parser.add_argument('--root_path', type=str,
                        default='./result_new/05Hz_03Hz_L_30_S_25_blocked_sim_1_to_120/align_img',
                        help='input image path')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()

    # GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train(args)
