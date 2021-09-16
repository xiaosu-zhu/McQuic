import torch
from torch import nn

from mcqc.layers.convs import conv1x1, conv3x3, deconv5x5
from mcqc.layers.gdn import GenDivNorm
from mcqc.layers.blocks import ResidualBlock, ResidualBlockUpsample, subPixelConv3x3, AttentionBlock


class Decoder(nn.Module):
    def __init__(self, inChannel=384, intermediateChannel=192):
        super().__init__()
        self._net = nn.Sequential(
            deconv5x5(inChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, 3, 2)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualDecoder(nn.Module):
    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualAttDecoder(nn.Module):
    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)

class ResidualBaseDecoder(nn.Module):
    def __init__(self, channel, groups):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class UpSampler(nn.Module):
    def __init__(self, channel, groups, outChannel=None):
        super().__init__()
        if outChannel is None:
            outChannel = channel
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualAttDecoderNew(nn.Module):
    def __init__(self, channel, groups, k):
        super().__init__()
        self._net = nn.Sequential(
            conv1x1(groups * k, channel, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            ResidualBlockUpsample(channel, channel, 2, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)
