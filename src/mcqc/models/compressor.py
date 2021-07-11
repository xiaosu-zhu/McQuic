from typing import Any
import sys

import torch
from torch import nn
from mcqc.layers.dropout import PointwiseDropout

from .encoder import ResidualEncoder, ResidualAttEncoder
from .decoder import ResidualDecoder, ResidualAttDecoder
from .contextModel import ContextModel
from .quantizer import AttentiveQuantizer


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
        self._groupDropout = PointwiseDropout(0.05, True)
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
        quantized = self._groupDropout(quantized)
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
