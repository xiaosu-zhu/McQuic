import torch
from torch import nn

from mcqc.layers.convs import conv3x3, conv5x5, pixelShuffle3x3
from mcqc.layers.gdn import GenDivNorm
from mcqc.layers.blocks import ResidualBlock, ResidualBlockUnShuffle, ResidualBlockWithStride, AttentionBlock


class BaseEncoder5x5(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(
            conv5x5(3, channel, groups=groups),
            GenDivNorm(channel),
            conv5x5(channel, channel, groups=groups),
            GenDivNorm(channel),
            conv5x5(channel, channel, groups=groups),
            GenDivNorm(channel),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class DownSampler5x5(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(
            conv5x5(channel, channel, groups=groups),
            GenDivNorm(channel),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class EncoderHead5x5(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        self._net = nn.Sequential(
            conv5x5(channel, channel, groups=groups),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)

class Director5x5(nn.Module):
    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(
            conv5x5(channel, channel, 1, groups=groups),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)




class ResidualBaseEncoder(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(
                ResidualBlockUnShuffle(3, channel),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockUnShuffle(channel, channel, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockUnShuffle(channel, channel, groups=groups),
            )
        else:
            self._net = nn.Sequential(
                ResidualBlockWithStride(3, channel, stride=2),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
            )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class DownSampler(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockUnShuffle(channel, channel, groups=groups),
            )
        else:
            self._net = nn.Sequential(
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
            )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class EncoderHead(nn.Module):
    def __init__(self, channel, groups, alias=False):
        super().__init__()
        if alias:
            self._net = nn.Sequential(
                ResidualBlock(channel, channel, groups=groups),
                pixelShuffle3x3(channel, channel, 2),
                AttentionBlock(channel, groups=groups)
            )
        else:
            self._net = nn.Sequential(
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups)
            )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)

class Director(nn.Module):
    def __init__(self, channel, groups, outChannels=None):
        super().__init__()
        if outChannels is None:
            outChannels = channel
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            conv3x3(channel, channel, stride=1, groups=groups)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)
