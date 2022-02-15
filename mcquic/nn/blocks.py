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


# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE./
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py


# NOTE: Slightly modified based on [CompVis/taming-transformers] for res-blocks, GroupNorm removed.

from math import sqrt
from typing import Union

import torch
from torch import nn

from mcquic.utils import ModuleRegistry
from .gdn import GenDivNorm, InvGenDivNorm
from .convs import MaskedConv2d, conv1x1, conv3x3, pixelShuffle3x3, pixelShuffle1x1


__all__ = [
    "ResidualBlockWithStride",
    "ResidualBlockShuffle",
    "ResidualBlock",
    "ResidualBlockMasked",
    "AttentionBlock",
    "NonLocalBlock"
]


class _residulBlock(nn.Module):
    def __init__(self, act1: nn.Module, conv1: nn.Conv2d, act2: nn.Module, conv2: nn.Conv2d, skip: Union[nn.Module, None]):
        super().__init__()
        self._branch = nn.Sequential(
            act1, conv1, act2, conv2
        )
        self._skip = skip

    def forward(self, x):
        identity = x
        out = self._branch(x)

        if self._skip is not None:
            identity = self._skip(x)

        out += identity
        return out


@ModuleRegistry.register
class ResidualBlockWithStride(_residulBlock):
    """Residual block with stride for down-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | Conv3s2   |  |
        | GDN       |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """
    def __init__(self, inChannels: int, outChannels: int, stride: int = 2, groups: int = 1):
        """Usage:
        ```python
            # A block performs 2x down-sampling
            block = ResidualBlockWithStride(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            stride (int): stride value (default: 2).
            groups (int): Group convolution (default: 1).
        """
        if stride != 1:
            skip = conv3x3(inChannels, outChannels, stride=stride)
        elif inChannels != outChannels:
            skip = conv1x1(inChannels, outChannels, stride=stride)
        else:
            skip = None
        super().__init__(
            nn.SiLU(),
            conv3x3(inChannels, outChannels, stride=stride),
            GenDivNorm(outChannels, groups=groups),
            conv3x3(outChannels, outChannels),
            skip)

@ModuleRegistry.register
class ResidualBlockShuffle(_residulBlock):
    """Residual block with PixelShuffle for up-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | PixShuf3  |  |
        | IGDN      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """
    def __init__(self, inChannels: int, outChannels: int, upsample: int = 2, groups: int = 1):
        """Usage:
        ```python
            # A block performs 2x up-sampling
            block = ResidualBlockShuffle(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            upsample (int): Up-sampling rate (default: 2).
            groups (int): Group convolution (default: 1).
        """
        super().__init__(
            nn.SiLU(),
            pixelShuffle3x3(inChannels, outChannels, upsample),
            InvGenDivNorm(outChannels, groups=groups),
            conv3x3(outChannels, outChannels),
            pixelShuffle3x3(inChannels, outChannels, upsample))


@ModuleRegistry.register
class ResidualBlock(_residulBlock):
    """Basic residual block.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | SiLU      |  |
        | Conv3s1   |  |
        | SiLU      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """
    def __init__(self, inChannels: int, outChannels: int, groups: int = 1):
        """Usage:
        ```python
            # A block with "same" feature map
            block = ResidualBlock(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            groups (int): Group convolution (default: 1).
        """
        if inChannels != outChannels:
            skip = conv1x1(inChannels, outChannels)
        else:
            skip = None
        super().__init__(
            nn.SiLU(),
            conv3x3(inChannels, outChannels),
            nn.SiLU(),
            conv3x3(outChannels, outChannels),
            skip)


@ModuleRegistry.register
class ResidualBlockMasked(_residulBlock):
    """A residual block with MaskedConv for causal inference.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | MConv5s1A |  |
        | LReLU     |  |
        | MConv5s1B |  |
        | LReLU     |  |
        | + <-------╯  |
        | Output       |
        +--------------+
    ```
    """
    def __init__(self, inChannels, outChannels, maskType: str = "A"):
        """Usage:
        ```python
            # First block
            block = ResidualBlockMasked(128, 128, "A")
            # Subsequent blocks
            block = ResidualBlockMasked(128, 128, "B")
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            maskType (str): Mask type of MaskedConv2D (default: "A").
        """
        if inChannels != outChannels:
            skip = MaskedConv2d(inChannels, outChannels, maskType=maskType, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode="zeros")
        else:
            skip = None
        super().__init__(
            nn.ReLU(),
            MaskedConv2d(inChannels, outChannels, maskType=maskType, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode="zeros"),
            nn.ReLU(),
            MaskedConv2d(outChannels, outChannels, maskType="B", bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode="zeros"),
            skip)


@ModuleRegistry.register
class AttentionBlock(nn.Module):
    """Self attention block.
    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Default structure:
    ```plain
        +----------------------+
        | Input ----┬--------╮ |
        | ResBlock  ResBlock | |
        | |         Sigmoid  | |
        | * <------ Mask     | |
        | Masked             | |
        | + <----------------╯ |
        | Output               |
        +----------------------+
    ```
    """
    def __init__(self, channel, groups=1):
        super().__init__()
        self._mainBranch = nn.Sequential(
            ResidualBlock(channel, channel, groups),
            ResidualBlock(channel, channel, groups),
            ResidualBlock(channel, channel, groups)
        )

        self._sideBranch = nn.Sequential(
            ResidualBlock(channel, channel, groups),
            ResidualBlock(channel, channel, groups),
            ResidualBlock(channel, channel, groups),
            conv1x1(channel, channel)
        )

    def forward(self, x):
        identity = x
        a = self._mainBranch(x)
        b = self._sideBranch(x)
        mask = torch.sigmoid(b)
        out = a * mask
        out += identity
        return out


@ModuleRegistry.register
class NonLocalBlock(nn.Module):
    def __init__(self, N, groups=1):
        super().__init__()
        self._c = N // 2
        self._q = conv1x1(N, N // 2, groups=groups)
        self._k = conv1x1(N, N // 2, groups=groups)
        self._v = conv1x1(N, N // 2, groups=groups)
        self._z = conv1x1(N // 2, N, groups=groups)

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        # x = self._position(x)
        hw = h*w
        scale = sqrt(hw)
        # [n, c/2, h, w]
        q = self._q(x).reshape(n, self._c, hw)
        k = self._k(x).reshape(n, self._c, hw)
        # [n, c/2, h, w] -> [n, hw, c/2]
        v = self._v(x).reshape(n, self._c, hw).permute(0, 2, 1)
        # [n, hw, hw]
        qkLogits = torch.matmul(q.transpose(-1, -2), k) / scale
        randomMask = torch.rand((n, hw, hw), device=qkLogits.device) < 0.1
        qkLogits = qkLogits.masked_fill(randomMask, -1e9)
        weights = torch.softmax(qkLogits, -1)
        # [n, hw, c/2] -> [n, c/2, h, w]
        z = torch.matmul(weights, v).permute(0, 2, 1).reshape(n, self._c, h, w)
        z = self._z(z)
        return x + z
