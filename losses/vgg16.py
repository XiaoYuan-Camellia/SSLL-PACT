# coding:utf-8
# @Time    : 2025/11/10 21:03
# @Author  : XiaoYuan
# @FileName: vgg16.py
# @description: VGG16 estimated features

import torch.nn as nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    def forward(self, x):
        features = []
        for name, module in self.features.named_children():
            x = module(x)
            if name in ['3', '8', '15', '22']:  # ReLU layers corresponding to relu1_2, relu2_2, relu3_3, relu4_3
                features.append(x)
        return features
