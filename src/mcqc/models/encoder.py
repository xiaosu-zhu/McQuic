import torch
from torch import nn

from mcqc.layers.convs import conv3x3, conv5x5
from mcqc.layers.gdn import GenDivNorm
from mcqc.layers.blocks import ResidualBlock, ResidualBlockDownSample, ResidualBlockWithStride, AttentionBlock


class Encoder(nn.Module):
    def __init__(self, intermediateChannel=192, outChannel=384):
        super().__init__()
        self._net = nn.Sequential(
            conv5x5(3, intermediateChannel),
            GenDivNorm(intermediateChannel),
            conv5x5(intermediateChannel, intermediateChannel),
            GenDivNorm(intermediateChannel),
            conv5x5(intermediateChannel, intermediateChannel),
            GenDivNorm(intermediateChannel),
            conv5x5(intermediateChannel, outChannel)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualEncoder(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(
                ResidualBlockDownSample(3, channel, 2),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, 2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, 2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
            )
        else:
            self._net = nn.Sequential(
                ResidualBlockWithStride(3, channel, stride=2),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
            )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualAttEncoder(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(
                ResidualBlockDownSample(3, channel),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
            )
        else:
            self._net = nn.Sequential(
                ResidualBlockWithStride(3, channel, stride=2),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
            )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)
