from typing import List, Tuple
import torch
from torch import nn

import mcquic
from mcquic.datasets.transforms import AlignedPadding
from mcquic.nn import pixelShuffle3x3
from mcquic.nn import ResidualBlock, ResidualBlockShuffle, ResidualBlockWithStride
from mcquic.nn.blocks import AttentionBlock
from mcquic.nn.convs import conv3x3
from mcquic.utils.specification import FileHeader, ImageSize

from .quantizer import BaseQuantizer, UMGMQuantizer


class BaseCompressor(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: BaseQuantizer):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._quantizer = quantizer
        self._qp = "-1"
        self._padding = AlignedPadding()

    @property
    def QuantizationParameter(self) -> str:
        return self._qp

    @QuantizationParameter.setter
    def QuantizationParameter(self, qp: str):
        self._qp = qp

    def forward(self, x: torch.Tensor):
        y = self._encoder(x)
        # [n, c, h, w], [n, m, h, w], [n, m, k, h, w]
        yHat, codes, logits = self._quantizer(y)
        xHat = self._decoder(yHat)
        return xHat, yHat, codes, logits

    # def count(self, x:torch.Tensor):
    #     y = self._encoder(x)
    #     self._quantizer.count(y)

    def reAssignCodebook(self) -> torch.Tensor:
        return self._quantizer.reAssignCodebook()

    def syncCodebook(self):
        return self._quantizer.syncCodebook()

    def readyForCoding(self):
        return self._quantizer.readyForCoding()

    @property
    def Freq(self):
        return self._quantizer.Freq

    @property
    def CodeUsage(self):
        return torch.cat(list((freq > 0).flatten() for freq in self._quantizer.Freq)).float().mean()

    def compress(self, x: torch.Tensor, cdfs: List[List[List[int]]]) -> Tuple[List[torch.Tensor], List[List[bytes]], List[FileHeader]]:
        n, c, h, w = x.shape

        x = self._padding(x)

        y = self._encoder(x)
        # codes: lv * [n, m, h, w]
        # binaries: List of binary, len = n, len(binaries[0]) = level
        codes, binaries, codeSizes = self._quantizer.compress(y, cdfs)
        header = [FileHeader(mcquic.__version__, self._qp, codeSize, ImageSize(height=h, width=w, channel=c)) for codeSize in codeSizes]
        return codes, binaries, header

    def decompress(self, binaries: List[List[bytes]], cdfs: List[List[List[int]]], headers: List[FileHeader]) -> torch.Tensor:
        yHat = self._quantizer.decompress(binaries, [header.CodeSize for header in headers], cdfs)
        restored = self._decoder(yHat)

        imageSize = headers[0].ImageSize

        H, W = restored.shape[-2], restored.shape[-1]

        h, w = imageSize.height, imageSize.width

        hCrop = H - h
        wCrop = W - w
        cropLeft = wCrop // 2
        cropRight = wCrop - cropLeft
        cropTop = hCrop // 2
        cropBottom = hCrop - cropTop

        if cropBottom == 0:
            cropBottom = -h
        if cropRight == 0:
            cropRight = -w

        return restored[..., cropTop:(-cropBottom), cropLeft:(-cropRight)]


class Compressor(BaseCompressor):
    def __init__(self, channel: int, m: int, k: List[int]):
        encoder = nn.Sequential(
            # convs.conv3x3(3, channel),
            conv3x3(3, channel, 2),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockWithStride(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockWithStride(channel, channel, groups=1),
            ResidualBlock(channel, channel, groups=1)
        )
        decoder = nn.Sequential(
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockShuffle(channel, channel, groups=1),
            AttentionBlock(channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            ResidualBlockShuffle(channel, channel, groups=1),
            ResidualBlock(channel, channel, groups=1),
            pixelShuffle3x3(channel, 3, 2)
        )
        quantizer = UMGMQuantizer(channel, m, k, {
            "latentStageEncoder": lambda: nn.Sequential(
                ResidualBlockWithStride(channel, channel, groups=1),
                # GroupSwishConv2D(channel, 3, groups=1),
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
            ),
            "quantizationHead": lambda: nn.Sequential(
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel, groups=1)
                # GroupSwishConv2D(channel, channel, groups=1)
            ),
            "latentHead": lambda: nn.Sequential(
                ResidualBlock(channel, channel, groups=1),
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel, groups=1)
            ),
            "restoreHead": lambda: nn.Sequential(
                AttentionBlock(channel, groups=1),
                ResidualBlock(channel, channel, groups=1),
                ResidualBlockShuffle(channel, channel, groups=1)
            ),
            "dequantizationHead": lambda: nn.Sequential(
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel, groups=1),
            ),
            "sideHead": lambda: nn.Sequential(
                AttentionBlock(channel, groups=1),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel, groups=1),
            ),
        })
        super().__init__(encoder, decoder, quantizer)
