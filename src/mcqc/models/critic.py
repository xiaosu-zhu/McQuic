from logging import Logger
import logging

import torch
from torch import nn

from mcqc.layers.convs import conv3x3, conv1x1
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample


class ResidualBNBlock(nn.Module):
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
        self.bn = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBNBlockWithStride(nn.Module):
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
        self.bn = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out



class SimpleCritic(nn.Module):
    def __init__(self, k):
        super().__init__()
        k = k[0]
        self._net = nn.Sequential(
            ResidualBNBlockWithStride(k, 256, stride=2),
            ResidualBNBlockWithStride(256, 256, stride=2),
            ResidualBNBlockWithStride(256, 256, stride=2),
            conv3x3(256, 1, stride=2)
        )

    def forward(self, logits: torch.Tensor):
        values = list()
        for logit in logits:
            values.append(self._net(logit).sum(axis=(-3, -2, -1)))
            return values

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator as in Pix2Pix
#         --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
#     """
#     def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if not use_actnorm:
#             norm_layer = nn.BatchNorm2d
#         else:
#             norm_layer = ActNorm
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func != nn.BatchNorm2d
#         else:
#             use_bias = norm_layer != nn.BatchNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [
#             nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.main = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.main(input)
