# coding:utf-8
# @Time    : 2025/11/15 11:32
# @Author  : XiaoYuan
# @FileName: FilteredBackProjection.py
# @description: Filtered Back-Projection (FBP) Algorithm

import torch
import numpy as np
import torch.nn as nn


class FilteredBackProjection(nn.Module):
    def __init__(self, R_ring, N_transducer, T_sample, x_vec, y_vec, angle_range=(0, 2 * torch.pi), theta_limit_deg=60):
        super().__init__()
        # parameters
        self.register_buffer("R_ring", torch.tensor(R_ring))
        self.N_transducer = N_transducer
        self.register_buffer("sampling_rate", torch.tensor(T_sample))
        self.H, self.W = len(x_vec), len(y_vec)
        self.register_buffer("x_vec", torch.tensor(x_vec).view(1, -1, 1))
        self.register_buffer("y_vec", torch.tensor(y_vec).view(1, 1, -1))
        angles = torch.linspace(angle_range[0], angle_range[1], N_transducer + 1)[:-1]
        self.register_buffer("angle_transducer", angles.view(-1, 1, 1))

        self.register_buffer("x_transducer", self.R_ring * torch.cos(self.angle_transducer))
        self.register_buffer("y_transducer", self.R_ring * torch.sin(self.angle_transducer))

        self.register_buffer("id_transducer",
                             torch.arange(N_transducer).view(-1, 1, 1).repeat(1, self.H, self.W))

        # Distance Computation
        dx = self.x_vec - self.x_transducer  # [N, H, W]
        dy = self.y_vec - self.y_transducer  # [N, H, W]
        self.register_buffer("distance_to_transducer", torch.sqrt(dx ** 2 + dy ** 2))

        # Directional Weight
        dist = torch.sqrt((self.x_transducer - self.x_vec) ** 2 + (self.y_transducer - self.y_vec) ** 2)
        w_vale = (R_ring ** 2 - (self.x_transducer * self.x_vec + self.y_transducer * self.y_vec)) / (R_ring * dist)
        cos_theta = torch.clamp(w_vale, min=0.)
        cos_threshold = torch.cos(torch.deg2rad(torch.tensor(theta_limit_deg)))
        self.w = torch.where(cos_theta > cos_threshold, cos_theta, torch.zeros_like(cos_theta)).cuda()

    def filter_wave(self, signogram):
        sensor_data = np.append(signogram, np.zeros((signogram.shape[0], 1)), axis=1)
        return -2 * (sensor_data[:, 1:] - sensor_data[:, :-1])

    def forward(self, sinogram, v0):
        assert sinogram.shape[0] == self.N_transducer, 'Invalid sinogram channels'
        # Filter operation
        sinogram = torch.from_numpy(self.filter_wave(sinogram)).cuda()
        # sinogram = torch.from_numpy(sinogram).cuda()

        # Calculate time index
        id_time = torch.round((self.distance_to_transducer / v0) * self.sampling_rate).long()
        id_time = id_time.clamp(0, sinogram.shape[1] - 1)
        return (sinogram[self.id_transducer, id_time] * self.w).sum(0)
