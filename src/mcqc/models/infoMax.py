from logging import Logger
import logging

import torch
from torch import nn

from mcqc.layers.convs import conv3x3, conv1x1
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample, ConvBlock
from mcqc.models.discriminator import ResidualBNBlock, ResidualBNBlockWithStride
from mcqc.models.encoder import ResidualEncoder


class Squeeze(nn.Module):
    def forward(self, x:torch.Tensor):
        return x[:, 0, 0, 0]


class ResidualBNEncoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ConvBlock(3, channel),
            ConvBlock(channel, channel),
            ConvBlock(channel, channel),
            ConvBlock(channel, channel),
            ConvBlock(channel, channel),
            ConvBlock(channel, channel),
            conv3x3(channel, channel, stride=2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        z = self._net(x)
        print(z.shape)
        exit()


class InfoMax(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._encoder = ResidualBNEncoder(channel)
        self._net = nn.Sequential(
            ConvBlock(2 * channel, 2 * channel), # 16
            ConvBlock(2 * channel, 2 * channel), # 8
            ConvBlock(2 * channel, 2 * channel), # 4
            conv1x1(2 * channel, 1, stride=1), # [n, 1, 4, 4]
            nn.AdaptiveAvgPool2d((1, 1)), # [n, 1, 1, 1]
            Squeeze()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = self._encoder(x)
        n = x.shape[0]
        # [n, 2d, h, w]
        # p(y|x)p(x)
        condition = torch.cat((z, y), 1)
        # [n, 2d, h, w]
        # shuffle y to get p(y)p(x)
        joint = torch.cat((z, y[torch.randperm(n)]), 1)
        # [2n, 2d, h, w]
        catted = torch.cat((condition, joint), 0)
        # [2n]
        predict = self._net(catted)
        return predict[:n], predict[n:]
