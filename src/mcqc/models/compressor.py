from typing import Any
import sys

import torch
from torch import nn
import torch.nn.functional as F
from cfmUtils.base import parallelFunction, Module
import storch
from mcqc.layers.dropout import ChannelWiseDropout, PointwiseDropout
from mcqc.models.encoderDecoder import EncoderDecoder, MLP

from mcqc.models.maskingModel import MaskingModel
from .encoder import ResidualEncoder, MultiScaleEncoder, ResidualAttEncoder
from .maskedLanguageModel import MaskedLangugeModel
from .decoder import ResidualDecoder, MultiScaleDecoder, ResidualAttDecoder
from .stackedAutoRegressive import StackedAutoRegressive
from .contextModel import ContextModel
from .quantizer import TransformerQuantizer, TransformerQuantizerStorch, AttentiveQuantizer, Quantizer
from mcqc.losses.structural import CompressionLoss
from mcqc.layers.blocks import L2Normalize


# class PQCompressor(nn.Module):
#     def __init__(self, m, k, channel, withAtt, withDropout, alias):
#         super().__init__()
#         self._k = k
#         self._m = m
#         self._encoder = ResidualAttEncoder(channel, alias) if withAtt else ResidualEncoder(channel, alias)
#         self._quantizer = Quantizer(dist.get_rank(), m, k, channel, withDropout, False, True)
#         self._decoder = ResidualAttDecoder(channel) if withAtt else ResidualDecoder(channel)

#     def forward(self, x: torch.Tensor, temp: float, e2e: bool):
#         latent = self._encoder(x)
#         quantized, codes, logits = self._quantizer(latent, temp, True)
#         restored = torch.tanh(self._decoder(quantized))
#         return restored, (quantized, latent), codes, logits


class PQCompressor(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias) if withAtt else ResidualEncoder(channel, groups, alias)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, withDropout, False, True) for _ in range(m))
        self._decoder = ResidualAttDecoder(channel, groups) if withAtt else ResidualDecoder(channel, groups)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        for quantizer, split in zip(self._quantizer, splits):
            q, c, l, wv = quantizer(split, temp)
            qs.append(q)
            codes.append(c.byte())
            logits.append(l)
        quantized = torch.cat(qs, 1)
        restored = torch.tanh(self._decoder(quantized))
        return restored, (quantized, latent), torch.stack(codes, 1), logits


class PQContextCompressor(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._k = k
        self._m = m
        self._encoder = ResidualEncoder(channel)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, False, True) for _ in range(m))
        self._decoder = ResidualDecoder(channel)
        self._context = nn.ModuleList(ContextModel(channel // m, 1, numLayers, channel // m, k) for _ in range(m))

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        predicts = list()
        targets = list()
        for quantizer, context, split in zip(self._quantizer, self._context, splits):
            q, c, l, wv = quantizer(split, temp)
            predict, target = context(q, c)
            predicts.append(predict)
            targets.append(target)
            qs.append(q)
            codes.append(c)
            logits.append(l)
        quantized = torch.cat(qs, 1)
        # [n, m, h, w]
        codes = torch.stack(codes, 1)
        restored = torch.tanh(self._decoder(quantized))
        return restored, codes, logits, predicts, targets


class PQSAGCompressor(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._k = k
        self._m = m
        self._encoder = ResidualEncoder(channel)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, False, True) for _ in range(m))
        self._decoder = ResidualDecoder(channel)
        self._context = MaskingModel(channel, m, numLayers, channel, k)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        for quantizer, split in zip(self._quantizer, splits):
            q, c, l, wv = quantizer(split, temp)
            qs.append(q)
            codes.append(c)
            logits.append(l)
        quantized = torch.cat(qs, 1)
        predicts, codes = self._context(quantized, codes)
        restored = torch.tanh(self._decoder(quantized))
        return restored, (quantized, latent), codes, logits, predicts, codes


class MultiScaleVQCompressor(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._encoder = MultiScaleEncoder(channel, nPreLayers, stage)
        self._quantizer = VQuantizer(k, channel, 0.1)
        self._decoder = MultiScaleDecoder(channel, nPreLayers, stage)

    def forward(self, x: torch.Tensor, temperature: float, hard: bool):
        latents = self._encoder(x)
        quantizeds, codes, zszq, codewords = self._quantizer(latents, temperature, hard)
        restored = torch.tanh(self._decoder(quantizeds))
        return restored, codes, latents, zszq, quantizeds, codewords # newLogits
