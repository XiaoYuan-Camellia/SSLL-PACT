# coding:utf-8
# @Time    : 2025/11/10 10:15
# @Author  : XiaoYuan
# @FileName: convEncoderAndVAE.py
# @description: convolutional Encoding and Latent Space Modeling via VAE

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoderAndVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoderAndVAE, self).__init__()

        self.conv_encoder = nn.Sequential(
            # 512 -> 256 (stride=2)
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # 256 -> 128 (stride=2)
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_output_size = 16 * 128 * 128

        self.fc1 = nn.Linear(self.conv_output_size, latent_dim)
        self.fc21 = nn.Linear(latent_dim, self.conv_output_size)  # Mean: revised to latent_dim
        self.fc22 = nn.Linear(latent_dim, self.conv_output_size)  # Log variance: revised to latent_dim

    def vae_encode(self, h1):
        h1 = h1.view(h1.size(0), -1)
        h1 = F.relu(self.fc1(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        b, _ , h, w = x.shape
        feature_maps = self.conv_encoder(x)
        mu, logvar = self.vae_encode(feature_maps)
        z = self.reparameterize(mu, logvar)
        z = z.view(b, -1, 128, 128)
        return z, mu, logvar
