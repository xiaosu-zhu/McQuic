from math import sqrt
from typing import Union, Tuple

import torch
from torch import nn

from mcqc import Consts
from mcqc.layers.positional import NPositionalEncoding2D

from .gdn import GenDivNorm
from .convs import conv1x1, conv3x3, conv5x5, subPixelConv3x3


class L2Normalize(nn.Module):
    def forward(self, x: torch.Tensor, dim: Union[int, Tuple[int]] = -1):
        norm = (x ** 2).sum(axis=dim, keepdim=True).sqrt()
        return x / norm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self._net = nn.Sequential(
            conv5x5(in_ch, out_ch, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self._net(x)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GenDivNorm(out_ch, beta_min=Consts.Eps)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subPixelConv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GenDivNorm(out_ch, inverse=True, beta_min=Consts.Eps)
        self.upsample = subPixelConv3x3(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class AttentionBlock(nn.Module):
    """Self attention block.
    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.
    Args:
        N (int): Number of channels)
    """

    def __init__(self, N):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class NonLocalBlock(nn.Module):
    def __init__(self, N):
        super().__init__()
        self._c = N // 2
        self._q = conv1x1(N, N // 2)
        self._k = conv1x1(N, N // 2)
        self._v = conv1x1(N, N // 2)
        self._z = conv1x1(N // 2, N)
        self._position = NPositionalEncoding2D(N, 120, 120)

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        x = self._position(x)
        hw = h*w
        scale = sqrt(hw)
        # [n, c/2, h, w]
        q = self._q(x).reshape(n, self._c, hw)
        k = self._k(x).reshape(n, self._c, hw)
        # [n, c/2, h, w] -> [n, hw, c/2]
        v = self._v(x).reshape(n, self._c, hw).permute(0, 2, 1)
        # [n, hw, hw]
        qkLogits = torch.matmul(q.transpose(-1, -2), k) / scale
        weights = torch.softmax(qkLogits, -1)
        # [n, hw, c/2] -> [n, c/2, h, w]
        z = torch.matmul(weights, v).permute(0, 2, 1).reshape(n, self._c, h, w)
        z = self._z(z)
        return x + z


class GlobalAttentionBlock(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, N):
        super().__init__()
        self._attention = NonLocalBlock(N)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(N, N)
        self.gdn = GenDivNorm(N, beta_min=Consts.Eps)

    def forward(self, x):
        identity = x
        out = self._attention(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        out += identity
        return out


class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel)
        )

    def forward(self, x):
        return self._net(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
        )

    def forward(self, x):
        return self._net(x)
