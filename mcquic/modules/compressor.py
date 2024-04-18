from typing import List, Tuple
import torch
from torch import nn

import mcquic
from mcquic.consts import Consts
from mcquic.datasets.transforms import AlignedPadding
from mcquic.nn import pixelShuffle3x3
from mcquic.nn import ResidualBlock, ResidualBlockShuffle, ResidualBlockWithStride
from mcquic.nn.blocks import AttentionBlock
from mcquic.nn.convs import conv3x3
from mcquic.utils.specification import FileHeader, ImageSize

from .quantizer import BaseQuantizer, UMGMQuantizer, NeonQuantizer, ResidualBackwardQuantizer


class BaseCompressor(nn.Module):
    def __init__(self, encoder: nn.Module, quantizer: BaseQuantizer, decoder: nn.Module):
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
        # [n, c, h, w], [n, m, h, w], [n, m, h, w, k]
        yHat, codes, logits = self._quantizer(y)
        xHat = self._decoder(yHat)
        return xHat, yHat, codes, logits

    def reAssignCodebook(self) -> torch.Tensor:
        return self._quantizer.reAssignCodebook()

    def syncCodebook(self):
        return self._quantizer.syncCodebook()

    @property
    def Codebooks(self):
        return self._quantizer.Codebooks

    @property
    def CDFs(self):
        return self._quantizer.CDFs

    @property
    def NormalizedFreq(self):
        return self._quantizer.NormalizedFreq

    @property
    def CodeUsage(self):
        return torch.cat(list((freq > Consts.Eps).flatten() for freq in self._quantizer.NormalizedFreq)).float().mean()

    def compress(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[bytes]], List[FileHeader]]:
        n, c, h, w = x.shape

        x = self._padding(x)

        y = self._encoder(x)
        # codes: lv * [n, m, h, w]
        # binaries: List of binary, len = n, len(binaries[0]) = level
        codes, binaries, codeSizes = self._quantizer.compress(y)
        header = [FileHeader(mcquic.__version__, self._qp, codeSize, ImageSize(height=h, width=w, channel=c)) for codeSize in codeSizes]
        return codes, binaries, header

    def decompress(self, binaries: List[List[bytes]], headers: List[FileHeader]) -> torch.Tensor:
        yHat = self._quantizer.decompress(binaries, [header.CodeSize for header in headers])
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
    def __init__(self, channel: int, m: int, k: List[int], permutationRate: float = 0.0):
        encoder = nn.Sequential(
            # convs.conv3x3(3, channel),
            conv3x3(3, channel, 2),
            ResidualBlock(channel, channel),
            ResidualBlockWithStride(channel, channel),
            AttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockWithStride(channel, channel),
            ResidualBlock(channel, channel)
        )
        decoder = nn.Sequential(
            ResidualBlock(channel, channel),
            ResidualBlockShuffle(channel, channel),
            AttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockShuffle(channel, channel),
            ResidualBlock(channel, channel),
            pixelShuffle3x3(channel, 3, 2)
        )
        quantizer = UMGMQuantizer(channel, m, k, permutationRate, {
            "latentStageEncoder": lambda: nn.Sequential(
                ResidualBlockWithStride(channel, channel),
                # GroupSwishConv2D(channel, 3),
                ResidualBlock(channel, channel),
                AttentionBlock(channel),
            ),
            "quantizationHead": lambda: nn.Sequential(
                ResidualBlock(channel, channel),
                AttentionBlock(channel),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel)
                # GroupSwishConv2D(channel, channel)
            ),
            "latentHead": lambda: nn.Sequential(
                ResidualBlock(channel, channel),
                AttentionBlock(channel),
                conv3x3(channel, channel)
                # convs.conv1x1(channel, channel)
            ),
            "restoreHead": lambda: nn.Sequential(
                AttentionBlock(channel),
                ResidualBlock(channel, channel),
                ResidualBlockShuffle(channel, channel)
            ),
            "dequantizationHead": lambda: nn.Sequential(
                AttentionBlock(channel),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel),
            ),
            "sideHead": lambda: nn.Sequential(
                AttentionBlock(channel),
                conv3x3(channel, channel),
                ResidualBlock(channel, channel),
            ),
        })
        super().__init__(encoder, quantizer, decoder)



class Neon(BaseCompressor):
    def __init__(self, channel: int, m: List[int], k: List[int], *_, **__):
        encoder = nn.Sequential(
            # convs.conv3x3(3, channel),
            conv3x3(3, 32),
            AttentionBlock(32),
            ResidualBlock(32, 320),
            ResidualBlock(320, 320),
            ResidualBlockWithStride(320, 320),
            ResidualBlock(320, 320),
            ResidualBlockWithStride(320, 320),
            ResidualBlock(320, 320),
            ResidualBlockWithStride(320, 320),
            AttentionBlock(320),
            ResidualBlock(320, 320),
            # ResidualBlock(320, 640),
            # ResidualBlockWithStride(320, 640),
            # ResidualBlock(640, 640),
            ResidualBlock(320, 640),
            ResidualBlock(640, 640),
            ResidualBlock(640, 320),
            ResidualBlock(320, 32),
            AttentionBlock(32),
        )
        decoder = nn.Sequential(
            AttentionBlock(32),
            ResidualBlock(32, 320),
            # AttentionBlock(32),
            ResidualBlock(320, 640),
            ResidualBlock(640, 640),
            ResidualBlock(640, 320),
            # ResidualBlockShuffle(640, 320),
            AttentionBlock(320),
            ResidualBlock(320, 320),
            ResidualBlockShuffle(320, 320),
            ResidualBlock(320, 320),
            ResidualBlockShuffle(320, 320),
            ResidualBlock(320, 320),
            ResidualBlockShuffle(320, 320),
            ResidualBlock(320, 320),
            ResidualBlock(320, 32),
            AttentionBlock(32),
            conv3x3(32, 3)
        )
        quantizer = ResidualBackwardQuantizer(m, k)
        super().__init__(encoder, quantizer, decoder)
