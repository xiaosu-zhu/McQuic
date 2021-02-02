from logging import Logger
import logging

import torch
from torch import nn

from mcqc.layers.convs import conv3x3
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample


class Discriminator(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            conv3x3(channel, channel, stride=2),
        )

    def forward(self, x: torch.Tensor):
        return self._net(x)


class FullDiscriminator(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockWithStride(3, channel, stride=2),
            ResidualBlock(channel, channel // 2),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel // 2, channel // 2, stride=2),
            ResidualBlock(channel // 2, channel // 4),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel // 4, channel // 4, stride=2),
            ResidualBlock(channel // 4, channel // 8),
            # AttentionBlock(channel),
            conv3x3(channel // 8, channel // 16, stride=2),
        )

    def forward(self, x: torch.Tensor):
        return self._net(x).sum(1)
