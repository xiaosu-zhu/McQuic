import torch
from torch import nn

from mcqc.nn.convs import pixelShuffle3x3, pixelShuffle5x5
from mcqc.nn.gdn import GenDivNorm, InvGenDivNorm
from mcqc.nn.blocks import ResidualBlock, AttentionBlock, ResidualBlockShuffle



class BaseDecoder5x5(nn.Module):
    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(
            pixelShuffle5x5(channel, channel, 2),
            InvGenDivNorm(channel),
            pixelShuffle5x5(channel, channel, 2),
            InvGenDivNorm(channel),
            pixelShuffle5x5(channel, 3, 2)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualBaseDecoder(nn.Module):
    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockShuffle(channel, channel, 2, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockShuffle(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            pixelShuffle3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class UpSampler(nn.Module):
    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockShuffle(channel, channel, 2, groups=groups),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class UpSampler5x5(nn.Module):
    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(
            pixelShuffle5x5(channel, channel, 2),
            GenDivNorm(channel)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)
