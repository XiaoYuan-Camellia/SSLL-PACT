# coding:utf-8
# @Time    : 2025/11/10 15:07
# @Author  : XiaoYuan
# @FileName: unetDecoder.py
# @description: Adopt U-Net as the decoding module.

import torch
import torch.nn as nn
from models.unet_parts import DoubleConv, Down, Up, OutConv


class UNetDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetDecoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)  #

        # Downsampling path: fixed 4 downsampling layers
        self.down1 = Down(32, 64)  # 128->64
        self.down2 = Down(64, 128)  # 64->32
        self.down3 = Down(128, 256)  # 32->16
        self.down4 = Down(256, 512)  # 16->8

        # Upsampling path: use skip connections
        self.up1 = Up(512, 256, bilinear)  # 8->16
        self.up2 = Up(256, 128, bilinear)  # 16->32
        self.up3 = Up(128, 64, bilinear)  # 32->64
        self.up4 = Up(64, 32, bilinear)  # 64->128


        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128->256
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 256->512
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.outc = OutConv(16, n_classes)

    def forward(self, x):

        x1 = self.inc(x)  # 128
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 32
        x4 = self.down3(x3)  # 16
        x5 = self.down4(x4)  # 8
        x = self.up1(x5, x4)  # 8->16
        x = self.up2(x, x3)  # 16->32
        x = self.up3(x, x2)  # 32->64
        x = self.up4(x, x1)  # 64->128
        # Finally upsample to a resolution of 512×512.
        x = self.final_upsample(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)
