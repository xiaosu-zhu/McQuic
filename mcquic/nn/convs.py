# Copyright 2020 InterDigital Communications, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py

from typing import Any

import torch
from torch import nn, Tensor


__all__ = [
    "MaskedConv2d",
    "conv3x3",
    "conv5x5",
    "deconv5x5",
    "pixelShuffle5x5",
    "pixelShuffle3x3",
    "conv1x1"
]


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.
    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.
    Inherits the same arguments as a `nn.Conv2d`. Use `maskType='A'` for the
    first layer (which also masks the "current pixel"), `maskType='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, maskType: str = "A", **kwargs: Any):
        """Masked Conv 2D.

        Args:
            args: Positional arguments for `nn.Conv2d`.
            maskType (str, optional): Mask type, if "A", current pixel will be masked, otherwise "B". Use "A" for first layer and "B" for successive layers. Defaults to "A".
            kwargs: Keyword arguments for `nn.Conv2d`.

        Usage:
        ```python
            # First layer
            conv = MaskedConv2d(3, 6, 3, maskType='A')
            # Subsequent layers
            conv = MaskedConv2d(6, 6, 3, maskType='B')
        ```

        Raises:
            ValueError: Mask type not in ["A", "B"].
        """
        super().__init__(*args, **kwargs)

        if maskType not in ("A", "B"):
            raise ValueError(f'Invalid `maskType` value "{maskType}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.shape
        self.mask[:, :, h // 2, w // 2 + (maskType == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)

def conv3x3(inChannels: int, outChannels: int, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    """A wrapper of 3x3 convolution with pre-calculated padding.

    Usage:
    ```python
        # A 3x3 conv with "same" feature map:
        conv = conv3x3(128, 128)
        # A 3x3 conv with halved feature map:
        conv = conv3x3(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=3, stride=stride, padding=1, padding_mode="reflect")

def conv5x5(inChannels: int, outChannels: int, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    """A wrapper of 5x5 convolution with pre-calculated padding.

    Usage:
    ```python
        # A 5x5 conv with "same" feature map:
        conv = conv5x5(128, 128)
        # A 5x5 conv with halved feature map:
        conv = conv5x5(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=5, stride=stride, padding=5 // 2, padding_mode="reflect")

def deconv5x5(inChannels: int, outChannels: int, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.ConvTranspose2d:
    """A wrapper of 5x5 TRANSPOSED convolution with pre-calculated padding.

    #### [CAUTION]: Use it only for proto-tests. Because PyTorch does not use CuDNN when stride > 1 (output_padding will be > 0).
    #### USE `subPixelConv5x5` instead if your want to do 2x up-sampling.

    Usage:
    ```python
        # A 5x5 deconv with "same" feature map:
        conv = deconv5x5(128, 128, 1)
        # A 5x5 deconv with doubled feature map:
        conv = deconv5x5(128, 128)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.ConvTranspose2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=5, stride=stride, padding=5 // 2, output_padding=stride - 1, padding_mode="zeros")

def pixelShuffle5x5(inChannels: int, outChannels: int, r: float = 1) -> nn.Conv2d:
    """A wrapper of 5x5 convolution and a 2x up-sampling by `PixelShuffle`.

    Usage:
    ```python
        # A 2x up-sampling with a 5x5 conv:
        conv = pixelShuffleConv5x5(128, 128, 2)
        # A 2x down-sampling with a 5x5 conv:
        conv = pixelShuffleConv5x5(128, 128, 0.5)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    if r < 1:
        r = int(1 / r)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels // (r ** 2), kernel_size=5, stride=1, padding=5 // 2, padding_mode="reflect"),
            nn.PixelUnshuffle(r)
        )
    else:
        r = int(r ** 2)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels * r, kernel_size=5, stride=1, padding=5 // 2, padding_mode="reflect"),
            nn.PixelShuffle(r)
        )


def pixelShuffle1x1(inChannels: int, outChannels: int, r: float = 1, groups: int = 1) -> nn.Conv2d:
    """A wrapper of 3x3 convolution and a 2x down-sampling by `PixelShuffle`.

    Usage:
    ```python
        # A 2x down-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 0.5)
        # A 2x up-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    if r < 1:
        r = int(1 / r)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels // (r ** 2), kernel_size=1, groups=groups),
            nn.PixelUnshuffle(r)
        )
    else:
        r = int(r)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels * (r ** 2), kernel_size=1, groups=groups),
            nn.PixelShuffle(r)
        )


def pixelShuffle3x3(inChannels: int, outChannels: int, r: float = 1, groups: int = 1) -> nn.Conv2d:
    """A wrapper of 3x3 convolution and a 2x down-sampling by `PixelShuffle`.

    Usage:
    ```python
        # A 2x down-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 0.5)
        # A 2x up-sampling with a 5x5 conv:
        conv = pixelShuffleConv3x3(128, 128, 2)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    if r < 1:
        r = int(1 / r)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels // (r ** 2), kernel_size=3, padding=1, groups=groups, padding_mode="reflect"),
            nn.PixelUnshuffle(r)
        )
    else:
        r = int(r)
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels * (r ** 2), kernel_size=3, padding=1, groups=groups, padding_mode="reflect"),
            nn.PixelShuffle(r)
        )

def conv1x1(inChannels: int, outChannels: int, stride: int = 1, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    """A wrapper of 1x1 convolution.

    Usage:
    ```python
        # A 1x1 conv with "same" feature map:
        conv = conv1x1(128, 128)
    ```

    Args:
        inChannels (int): Channels of input.
        outChannels (int): Channels of output.
        stride (int, optional): Stride. Defaults to 1.
        bias (bool, optional): Bias. Defaults to True.
        groups (int, optional): Group convolution. Defaults to 1.

    Returns:
        nn.Module: A Conv layer.
    """
    return nn.Conv2d(inChannels, outChannels, bias=bias, groups=groups, kernel_size=1, stride=stride, padding_mode="reflect")
