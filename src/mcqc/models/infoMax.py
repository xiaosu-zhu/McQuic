from logging import Logger
import logging
from dreinq.layers.layerGroup import LayerGroup

import torch
from torch import nn

from dreinq.layers.incepBlock import IncepBlock

from mcqc.layers.convs import conv3x3, conv1x1
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample, ConvBlock
from mcqc.models.discriminator import ResidualBNBlock, ResidualBNBlockWithStride
from mcqc.models.encoder import ResidualEncoder


class Squeeze(nn.Module):
    def forward(self, x:torch.Tensor):
        return x[:, :, 0, 0]


class ResidualBNEncoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBNBlock(3, channel),
            ConvBlock(channel, channel),
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel),
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel),
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel), # 16
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel), # 8
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel), # 4
            ResidualBNBlock(channel, channel),
            conv1x1(channel, channel),
            nn.AdaptiveAvgPool2d((1, 1)), # [n, c, 1, 1]
            Squeeze()
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class InfoMax(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._estimatorOverY = ResidualBNEncoder(channel)
        self._estimatorOverT = nn.Sequential(
            ConvBlock(channel, channel), # 16
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel), # 8
            ResidualBNBlock(channel, channel),
            ConvBlock(channel, channel), # 4
            ResidualBNBlock(channel, channel),
            conv1x1(channel, channel), # [n, c, 4, 4]
            nn.AdaptiveAvgPool2d((1, 1)), # [n, c, 1, 1]
            Squeeze()
        )
        self._final = nn.Sequential(
            IncepBlock(2 * channel, 0.1, norm="gn"),
            LayerGroup(2 * channel, channel, 0.1, nn.ReLU, "gn"),
            IncepBlock(channel, 0.1, norm="gn"),
            IncepBlock(channel, 0.1, norm="gn"),
            nn.Linear(channel, 1)
        )

    def forward(self, y: torch.Tensor, t: torch.Tensor):
        # [n, c]
        zy = self._estimatorOverY(y)
        zt = self._estimatorOverT(t)
        n = y.shape[0]
        # [n, 2c]
        # p(y|x)p(x)
        condition = torch.cat((zy, zt), 1)
        # [n, 2c]
        # shuffle y to get p(y)p(x)
        joint = torch.cat((zy[torch.randperm(n)], zt[torch.randperm(n)]), 1)
        # [2n, 2c]
        catted = torch.cat((condition, joint), 0)
        # [2n]
        predict = self._final(catted)[:, 0]
        return predict[:n], predict[n:]
