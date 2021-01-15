from logging import Logger
import logging

import torch
from torch import nn

from mcqc.layers.convs import conv3x3
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample


class Encoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockWithStride(6, channel, stride=2),
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
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, channel, scale):
        super().__init__()
        # 1/4, 1/4
        self._preProcess = nn.Sequential(
            ResidualBlockWithStride(6, channel, stride=2),
            ResidualBlock(channel, channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # ResidualBlockWithStride(channel, channel, stride=2),
            # ResidualBlock(channel, channel)
        )
        # DownSample -> 1/2, 1/2
        self._scales = nn.ModuleList([DownSample(channel) for _ in range(scale)])
        # conv -> 1/2, 1/2
        self._postProcess = nn.ModuleList([conv3x3(channel, channel, stride=2) for _ in range(scale)])

    def forward(self, x: torch.Tensor):
        # 1/4, 1/4
        x = self._preProcess(x)
        results = list()
        for scale, process in zip(self._scales, self._postProcess):
            # each loop: x / 2, result = x / 2 / 2
            x = scale(x)
            results.append(process(x))
        return results
