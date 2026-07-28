# coding:utf-8
# @Time    : 2025/11/8 17:46
# @Author  : XiaoYuan
# @FileName: kl_loss.py
# @description: Kullback-Leibler Divergence Loss (KL loss)

import torch
import torch.nn as nn


class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()

    def forward(self, mu, log_var):
        # Calculate per-sample KL divergence,sum along feature dimension.
        kl_loss_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss_per_sample.mean()
        return kl_loss
