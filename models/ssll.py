# coding:utf-8
# @Time    : 2025/11/8 17:38
# @Author  : XiaoYuan
# @FileName: ssll.py
# @description: Overall Framework of Self-Supervised Latent Space Learning (SSLL)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from losses.kl_loss import KL
from losses.ssim_loss import SSIM_LOSS
from losses.vgg16 import VGG16FeatureExtractor
from models.convEncoderAndVAE import ConvEncoderAndVAE
from models.denseLatentSpace import DenseLatentSpace
from models.unetDecoder import UNetDecoder


class SelfSupervisedLatentSpaceLearn(nn.Module):
    def __init__(self, latent_dim=2048, in_channle=16, image_batch=4, lr=1e-3):
        super(SelfSupervisedLatentSpaceLearn, self).__init__()
        # define model
        self.conv_encoder_vae = ConvEncoderAndVAE(latent_dim=latent_dim)
        self.dense_latent_space = DenseLatentSpace(in_channle=in_channle, image_batch=image_batch)
        self.unet_decoder = UNetDecoder(n_channels=in_channle, n_classes=1)

        # define optimizer
        self.optimizer = Adam([
            {"params": self.conv_encoder_vae.parameters()},
            {"params": self.dense_latent_space.parameters()},
            {"params": self.unet_decoder.parameters()}], lr=lr)

        # define loss functions
        self.kl_loss = KL()
        self.ssim_loss_fn = SSIM_LOSS()
        self.vgg_16 = VGG16FeatureExtractor()

    def forward_compute(self, x):
        """
            computer forward process
        Args:
            x: image information

        Returns:
            img, mu, log_var
        """
        # encode patient space
        z, mu, log_var = self.conv_encoder_vae(x)  # Latent space, latent mean (mu) and log variance (log_var)
        # feature fusion
        feature_vector = self.dense_latent_space(z)
        # feature decode
        img = self.unet_decoder(feature_vector)
        return img, mu, log_var


    def Fro_LOSS(self, x):
        return torch.norm(x, p='fro') / (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]) ** 0.5

    def features_grad(self, features):
        kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                               [1 / 8, -1, 1 / 8],
                               [1 / 8, 1 / 8, 1 / 8]]).cuda().float()
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(features.shape[1], 1, 1, 1)

        padding = kernel.shape[-1] // 2
        fgs = F.conv2d(features, kernel, padding=padding, groups=features.shape[1])
        return fgs

    def compute_single_source_weights(self, SOURCE, c=1.0):
        """
            Compute VGG-based weights for single source

        Args:
            SOURCE: Source image with tensor shape (batch, channel, height, width): (b, 1, 512, 512)
            c: normalization constant

        Returns:
            weights: Weight tensor: shape (b, 1)
        """
        SOURCE_VGG_in = F.interpolate(SOURCE.repeat(1, 3, 1, 1), size=(224, 224), mode='nearest')

        SOURCE_FEAS = self.vgg_16(SOURCE_VGG_in)

        # Calculate feature gradients and average them
        ws_list = []
        for i in range(len(SOURCE_FEAS)):
            # Calculate mean value of squared feature gradients
            grad = self.features_grad(SOURCE_FEAS[i])
            m = torch.mean(torch.square(grad), dim=[1, 2, 3])
            ws_list.append(m.unsqueeze(-1))

        ws = torch.cat(ws_list, dim=-1)
        s = torch.mean(ws, dim=-1) / c
        weights = torch.softmax(s, dim=-1)

        return weights.unsqueeze(-1)  # reshape tensor from shape (b) to (b, 1)

    def compute_losses(self, source_images):
        b, c, h, w = source_images.shape
        fused_clear_img, mu, log_var = self.forward_compute(source_images)

        re_fused_clear_img = fused_clear_img.repeat(b, 1, 1, 1)  # repeat the corresponding images

        # SSIM loss
        ssim_loss = self.ssim_loss_fn(source_images, re_fused_clear_img)

        # MSE loss (Frobenius norm)
        mse_loss = self.Fro_LOSS(re_fused_clear_img - source_images)

        # VGG feature extraction and gradient computation
        s = self.compute_single_source_weights(source_images)

        ssim_loss = torch.mean(s * ssim_loss)
        mse_loss = torch.mean(s * mse_loss)
        rec_loss = ssim_loss + 10 * mse_loss

        return rec_loss, fused_clear_img, mu, log_var

    def forward(self, x, beta):
        """
            Forward propagation
        Args:
            x: input image
            beta: controller parameter

        Returns:
            fused_clear_img: fuse clear image
            rec_loss: reconstruct loss
            kl_loss: Kullback-Leibler(KL) Divergence Loss
        """
        self.optimizer.zero_grad()
        # Compute the loss function
        rec_loss, fused_clear_img, mu, log_var = self.compute_losses(x)

        # update
        kl_loss = self.kl_loss(mu, log_var)
        total_loss = rec_loss * (1 - beta) + kl_loss * beta

        total_loss.backward()
        self.optimizer.step()
        return fused_clear_img, rec_loss, kl_loss, total_loss

    def predict(self, x):
        fused_clear_img, mu, log_var = self.forward_compute(x)
        return fused_clear_img.squeeze().squeeze()
