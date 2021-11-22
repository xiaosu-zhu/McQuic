from typing import Any

import torch
from torch import nn, Tensor


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.shape
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch, out_ch, stride=1, bias=True, groups=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, groups=groups, kernel_size=3, stride=stride, padding=1, padding_mode="reflect")

def conv5x5(in_ch, out_ch, stride=2, bias=True, groups=1):
    """5x5 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, groups=groups, kernel_size=5, stride=stride, padding=5 // 2, padding_mode="reflect")

def deconv5x5(in_ch, out_ch, stride=2, bias=True, groups=1):
    """ !!!USE IT CAREFULLY, output_padding cause it very slow!!!
    USE deconv5x5Up instead if your want to do 2x up-sampling.
    5x5 convolution with padding."""
    return nn.ConvTranspose2d(in_ch, out_ch, bias=bias, groups=groups, kernel_size=5, stride=stride, padding=5 // 2, output_padding=stride - 1, padding_mode="zeros")

def deconv5x5Up(in_ch, out_ch, bias=True, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, bias=bias, groups=groups, kernel_size=5, stride=1, padding=5 // 2, padding_mode="reflect"),
            nn.PixelShuffle(2)
    )

def subPixelConv3x3(in_ch, out_ch, r=1, groups=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, groups=groups, kernel_size=3, padding=1, padding_mode="reflect"), nn.PixelShuffle(r)
    )

def superPixelConv3x3(in_ch, out_ch, r=1, groups=1):
    """3x3 super-pixel convolution for down-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch // (r ** 2), groups=groups, kernel_size=3, padding=1, padding_mode="reflect"), nn.PixelUnshuffle(r)
    )

def conv1x1(in_ch, out_ch, stride=1, bias=True, groups=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, groups=groups, kernel_size=1, stride=stride, padding_mode="reflect")
