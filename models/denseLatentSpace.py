# coding:utf-8
# @Time    : 2025/11/10 15:07
# @Author  : XiaoYuan
# @FileName: denseLatentSpace.py
# @description: Dense latent-space fusion module


import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_INIT_STDDEV = 0.05
n = 44


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_lrelu=True, dense=False):
        super(Conv2d, self).__init__()
        self.use_lrelu = use_lrelu
        self.dense = dense
        self.padding = kernel_size // 2  # For 'SAME' padding equivalent

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=self.padding, bias=True)

        # Initialize weights
        nn.init.trunc_normal_(self.conv.weight, std=WEIGHT_INIT_STDDEV)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # Reflection padding (different from conv padding)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # Remove the padding we added for convolution since we're using custom reflection padding
        conv = self.conv
        conv.padding = 0
        out = conv(x_padded)
        conv.padding = (self.padding, self.padding)  # Restore

        if self.use_lrelu:
            out = F.leaky_relu(out, 0.2)

        if self.dense:
            out = torch.cat([out, x], dim=1)

        return out


class Encoder(nn.Module):
    def   __init__(self, in_channle=2):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            Conv2d(in_channle, n, 3, use_lrelu=True, dense=False),  # conv1_1
            Conv2d(n, n, 3, use_lrelu=True, dense=True),  # dense_block_conv1
            Conv2d(n * 2, n, 3, use_lrelu=True, dense=True),  # dense_block_conv2
            Conv2d(n * 3, n, 3, use_lrelu=True, dense=True),  # dense_block_conv3
            Conv2d(n * 4, n, 3, use_lrelu=True, dense=True),  # dense_block_conv4
            Conv2d(n * 5, n, 3, use_lrelu=True, dense=True)  # dense_block_conv5
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channle=1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            Conv2d(n * 6, 128, 3, use_lrelu=True),  # conv2_1
            Conv2d(128, 64, 3, use_lrelu=True),  # conv2_2
            Conv2d(64, 32, 3, use_lrelu=True),  # conv2_3
            Conv2d(32, out_channle, 3, use_lrelu=False)  # conv2_4
        ])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == len(self.layers) - 1:  # Last layer
                out = torch.tanh(out) / 2 + 0.5
        return out


class DenseLatentSpace(nn.Module):
    def __init__(self, in_channle=1, image_batch=1):
        super(DenseLatentSpace, self).__init__()
        self.encoder = Encoder(in_channle=in_channle * image_batch)
        self.decoder = Decoder(out_channle=in_channle)

    def forward(self, I1):
        img = torch.chunk(I1, I1.shape[0], dim=0)
        img = torch.cat(img, dim=1)
        code = self.encoder(img)
        generated_img = self.decoder(code)
        return generated_img
