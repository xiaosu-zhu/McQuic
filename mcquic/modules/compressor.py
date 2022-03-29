from copy import deepcopy
from typing import List, Tuple

import torch
from torch import nn

import mcquic
from mcquic.config import Config
from mcquic.datasets.transforms import AlignedPadding
from mcquic.nn import pixelShuffle3x3
from mcquic.nn import ResidualBlock, ResidualBlockShuffle, ResidualBlockWithStride
from mcquic.nn.blocks import AttentionBlock
from mcquic.nn.convs import conv3x3
from mcquic.utils.specification import FileHeader, ImageSize
from mcquic.rans import RansEncoder, RansDecoder

from .quantizer import BaseQuantizer, UMGMQuantizer


class BaseCompressor(nn.Module):
    cdfs: List[List[List[int]]]
    qp: str
    config: str
    version: str
    k: List[int]
    def __init__(self, encoder: nn.Module, quantizer: BaseQuantizer, decoder: nn.Module):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._quantizer = quantizer
        self.qp = "-1"
        self._padding = AlignedPadding()

    @property
    def QuantizationParameter(self) -> str:
        return self.qp

    @QuantizationParameter.setter
    def QuantizationParameter(self, qp: str):
        self.qp = qp

    def forward(self, x: torch.Tensor):
        if torch.jit.is_scripting():
            codes, size = self.encode(x)
            xHat = self.decode(codes, size)
            return xHat
        else:
            y = self._encoder(x)
            # [n, c, h, w], [n, m, h, w], [n, m, h, w, k]
            yHat, codes, logits = self._quantizer(y)
            xHat = self._decoder(yHat)
            return xHat, yHat, codes, logits

    def reAssignCodebook(self) -> torch.Tensor:
        return self._quantizer.reAssignCodebook()

    def syncCodebook(self):
        return self._quantizer.syncCodebook()

    @torch.jit.ignore
    def readyForCoding(self):
        return self._quantizer.readyForCoding()

    def exportForJIT(self, device, config: Config):
        self.to(device)
        self.config = config.dump()
        self.version = mcquic.__version__
        qp = config.Model.Params["m"]
        self.qp = f"qp_{qp}_{config.Train.Target}"
        self.k = config.Model.Params["k"]
        with self.readyForCoding() as cdfs:
            self.cdfs = deepcopy(cdfs)
        return torch.jit.freeze(torch.jit.script(self), ["encode", "decode", "cdfs", "qp", "_encoder", "_decoder", "_quantizer", "_padding", "config", "version", "k"])

    @property
    @torch.jit.unused
    def Freq(self):
        return self._quantizer.Freq

    @property
    @torch.jit.unused
    def CodeUsage(self):
        return torch.cat(list((freq > 0).flatten() for freq in self._quantizer.Freq)).float().mean()

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        _, _, h, w = x.shape
        x = self._padding(x)
        y = self._encoder(x)
        # codes: lv * [n, m, h, w]
        codes = self._quantizer.encode(y)
        return codes, (h, w)

    @torch.jit.ignore
    def compress(self, encoder: RansEncoder, codes: List[torch.Tensor], size: Tuple[int, int], cdfs: List[List[List[int]]]) -> Tuple[List[List[bytes]], List[FileHeader]]:
        # binaries: List of binary, len = n, len(binaries[0]) = level
        binaries, codeSizes = self._quantizer.compress(encoder, codes, cdfs)
        h, w = size
        header = [FileHeader(mcquic.__version__, self.qp, codeSize, ImageSize(height=h, width=w, channel=3)) for codeSize in codeSizes]
        return binaries, header

    @torch.jit.ignore
    def decompress(self, decoder: RansDecoder, binaries: List[List[bytes]], cdfs: List[List[List[int]]], headers: List[FileHeader]) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        codes = self._quantizer.decompress(decoder, binaries, [header.CodeSize for header in headers], cdfs)
        imageSize = (headers[0].ImageSize.height, headers[0].ImageSize.width)
        return codes, imageSize

    @torch.jit.export
    def decode(self, codes: List[torch.Tensor], imageSize: Tuple[int, int]) -> torch.Tensor:
        yHat = self._quantizer.decode(codes)
        restored = self._decoder(yHat)

        H, W = restored.shape[-2], restored.shape[-1]

        h, w = imageSize

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
        super().__init__(encoder, quantizer, decoder)
