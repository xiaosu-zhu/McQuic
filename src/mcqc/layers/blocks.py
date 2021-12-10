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

from math import sqrt
from typing import Union

import torch
from torch import nn
from vlutils.base import Registry

from .gdn import GenDivNorm
from .convs import MaskedConv2d, conv1x1, conv3x3, conv5x5, pixelShuffle3x3


class BlockFactory(Registry):
    """A factory of blocks.
    """


# NOTE: Based on [CompVis/taming-transformers]
class _baseAct(nn.Module):
    def __init__(self, groups: int, channels: int):
        super().__init__()
        self._norm = nn.GroupNorm(groups, channels, 1e-6, True)
        self._act = nn.SiLU(True)
    def forward(self, x):
        return self._act(self._norm(x))

# NOTE: Slightly modified based on [CompVis/taming-transformers]
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


@BlockFactory.register("ResidualWithStride")
class ResidualBlockWithStride(_residulBlock):
    """Residual block with stride for down-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | GroupN    |  |
        | SiLU      |  |
        | Conv3s2   |  |
        | GroupN    |  |
        | SiLU      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        | GDN          |
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
            _baseAct(groups, inChannels),
            conv3x3(inChannels, outChannels, stride=stride),
            GenDivNorm(outChannels),
            conv3x3(outChannels, outChannels),
            skip)


@BlockFactory.register("ResidualBlockUnShuffle")
class ResidualBlockUnShuffle(_residulBlock):
    """Residual block with PixelUnShuffle for down-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | GroupN    |  |
        | SiLU      |  |
        | Conv3s1   |  |
        | PixUnShuf |  |
        | GroupN    |  |
        | SiLU      |  |
        | Conv3s1   |  |
        | + <-------╯  |
        | Output       |
        | GDN          |
        +--------------+
    ```
    """
    def __init__(self, inChannels: int, outChannels: int, downsample: int = 2, groups: int = 1):
        """Usage:
        ```python
            # A block performs 2x down-sampling
            block = ResidualBlockUnShuffle(128, 128)
        ```
        Args:
            inChannels (int): Channels of input.
            outChannels (int): Channels of output.
            downsample (int): Down-sampling rate (default: 2).
            groups (int): Group convolution (default: 1).
        """
        super().__init__(
            _baseAct(groups, inChannels),
            pixelShuffle3x3(inChannels, outChannels, 1 / downsample),
            GenDivNorm(outChannels),
            conv3x3(outChannels, outChannels),
            pixelShuffle3x3(inChannels, outChannels, 1 / downsample))


@BlockFactory.register("ResidualBlockShuffle")
class ResidualBlockShuffle(_residulBlock):
    """Residual block with PixelShuffle for up-sampling.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | Conv3s1   |  |
        | PixShuf   |  |
        | LReLU     |  |
        | Conv3s1   |  |
        | IGDN      |  |
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
            _baseAct(groups, inChannels),
            pixelShuffle3x3(inChannels, outChannels, upsample),
            GenDivNorm(outChannels, inverse=True),
            conv3x3(outChannels, outChannels),
            pixelShuffle3x3(inChannels, outChannels, upsample))


@BlockFactory.register("ResidualBlock")
class ResidualBlock(_residulBlock):
    """Basic residual block.

    Default structure:
    ```plain
        +--------------+
        | Input ----╮  |
        | Conv3s1   |  |
        | LReLU     |  |
        | Conv3s1   |  |
        | LReLU     |  |
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
            _baseAct(groups, inChannels),
            conv3x3(inChannels, outChannels),
            _baseAct(groups, outChannels),
            conv3x3(outChannels, outChannels),
            skip)


@BlockFactory.register("ResidualBlockMasked")
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
            MaskedConv2d(outChannels, outChannels, maskType="B", bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode="zeros"),
            conv3x3(outChannels, outChannels),
            skip)


@BlockFactory.register("AttentionBlock")
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
    def __init__(self, N, groups=1):
        super().__init__()

        class _residualUnit(nn.Module):
            """Simplified residual unit."""
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2, groups=groups),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2, groups=groups),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N, groups=groups),
                )
                self.relu = nn.ReLU(inplace=True)
            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(_residualUnit(), _residualUnit(), _residualUnit())

        self.conv_b = nn.Sequential(
            _residualUnit(),
            _residualUnit(),
            _residualUnit(),
            conv1x1(N, N, groups=groups),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        mask = torch.sigmoid(b)
        out = a * mask
        out += identity
        return out


@BlockFactory.register("NonLocalBlock")
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
