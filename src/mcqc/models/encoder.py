import torch
from torch import nn

from mcqc.layers.convs import conv1x1, conv3x3, conv5x5, superPixelConv3x3
from mcqc.layers.gdn import GenDivNorm
from mcqc.layers import ResidualBlock, ResidualBlockDownSample, ResidualBlockWithStride, AttentionBlock




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
    def __init__(self, channel, groups, outChannel=None):
        super().__init__()
        if outChannel is None:
            outChannel = channel
        self._net = nn.Sequential(
            conv5x5(channel, channel, 1, groups=groups),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)

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


class ResidualBaseEncoder(nn.Module):
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
                ResidualBlockDownSample(channel, channel, groups=groups),
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
                superPixelConv3x3(channel, channel, 2, groups=groups),
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
    def __init__(self, channel, groups, outChannel=None):
        super().__init__()
        if outChannel is None:
            outChannel = channel
        self._net = nn.Sequential(
            AttentionBlock(channel, groups=groups),
            ResidualBlock(channel, channel, groups=groups),
            conv3x3(channel, channel, stride=1, groups=groups)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualAttEncoderNew(nn.Module):
    def __init__(self, channel, groups, k, alias=False):
        super().__init__()

        if alias:
            net = [ResidualBlockDownSample(3, channel),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockDownSample(channel, channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
                conv1x1(channel, groups * k, groups=groups)]
        else:
            net = [ResidualBlockWithStride(3, channel, stride=2),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                ResidualBlockWithStride(channel, channel, stride=2, groups=groups),
                ResidualBlock(channel, channel, groups=groups),
                conv3x3(channel, channel, stride=2, groups=groups),
                AttentionBlock(channel, groups=groups),
                conv1x1(channel, groups * k, groups=groups)]
        self._net = nn.Sequential(*net)
        self._quantizer = QuantizeBlock(groups, k)

    def forward(self, x: torch.Tensor, temperature: float):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        logit = self._net(x)
        code, logit = self._quantizer(logit, temperature)
        return code, logit
