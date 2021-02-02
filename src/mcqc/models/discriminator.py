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
