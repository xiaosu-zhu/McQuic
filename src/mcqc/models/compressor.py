from typing import Any
import sys

import torch
from torch import nn
import torch.nn.functional as F
from cfmUtils.base import parallelFunction, Module
import storch

from .encoder import Encoder, MultiScaleEncoder, TransformerEncoder
from .decoder import Decoder, MultiScaleDecoder, TransformerDecoder
from .quantizer import TransformerQuantizer, VQuantizer, TransformerQuantizerRein, TransformerQuantizerStorch, AttentiveQuantizer
from mcqc.losses.structural import CompressionLoss


class MultiScaleCompressor(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._encoder = MultiScaleEncoder(channel, nPreLayers, 1)
        self._quantizer = TransformerQuantizer(k, channel, 0.1)
        self._decoder = MultiScaleDecoder(channel, nPreLayers, 1)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latents = self._encoder(x)
        quantizeds, codes, logits = self._quantizer(latents, temp, True)
        if e2e is None:
            restored = torch.tanh(self._decoder(latents))
        elif not e2e:
            mixeds = list()
            for latent, q in zip(latents, quantizeds):
                mixeds.append((q - latent).detach() + latent)
            restored = torch.tanh(self._decoder(mixeds))
        else:
            restored = torch.tanh(self._decoder(quantizeds))
        return restored, codes, latents, logits, quantizeds


class MultiScaleCompressorSplitted(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._preEncoder = MultiScaleEncoder(channel, nPreLayers, 1)
        self._transEncoder = TransformerEncoder(k, channel)
        self._quantizer = AttentiveQuantizer(k, channel, 0.1)
        self._decoder = nn.Sequential(TransformerDecoder(channel), MultiScaleDecoder(channel, nPreLayers, 1))

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latents = self._preEncoder(x)
        latents, predicts = self._transEncoder(latents, self._quantizer.getCodebook())
        quantizeds, codes, logits = self._quantizer(latents, temp, True)
        if e2e is None:
            restored = torch.tanh(self._decoder(latents))
        elif not e2e:
            mixeds = list()
            for latent, q in zip(latents, quantizeds):
                mixeds.append((q - latent).detach() + latent)
            restored = torch.tanh(self._decoder(mixeds))
        else:
            restored = torch.tanh(self._decoder(quantizeds))
        return restored, codes, latents, (predicts, logits), quantizeds

class MultiScaleCompressorExp(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._encoder = MultiScaleEncoder(channel, nPreLayers, 1)
        self._quantizer = TransformerQuantizer(k, channel, 0.1)
        self._decoder = MultiScaleDecoder(channel, nPreLayers, 1)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        if e2e:
            latents = self._encoder(x)
        else:
            with torch.no_grad():
                latents = self._encoder(x)
        quantizeds, codes, logits = self._quantizer(latents, temp, True)
        if e2e:
            restored = torch.tanh(self._decoder(latents))
        elif e2e is None:
            with torch.no_grad():
                restored = torch.tanh(self._decoder(latents))
        else:
            restored = torch.tanh(self._decoder(quantizeds))
        return restored, codes, latents, logits, quantizeds


class MultiScaleCompressorStorch(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._encoder = MultiScaleEncoder(channel, nPreLayers, 1)
        self._quantizer = TransformerQuantizerStorch(k, channel, 0.1)
        self._decoder = MultiScaleDecoder(channel, nPreLayers, 1)

    def forward(self, x: torch.Tensor, temp: float, transform: bool):
        latents = self._encoder(x)
        quantizeds, codes, logits = self._quantizer(latents, temp, transform)
        restored = torch.tanh(self._decoder(quantizeds))
        # clipped = restored.clamp(-1.0, 1.0)
        # restored = (clipped - restored).detach() + restored
        return restored, codes, latents, logits, quantizeds


class MultiScaleCompressorRein(nn.Module):
    def __init__(self, k , channel, nPreLayers):
        super().__init__()
        stage = len(k)
        self._encoder = MultiScaleEncoder(channel, nPreLayers, stage)
        self._quantizer = TransformerQuantizerRein(k, channel, 0.1)
        self._decoder = MultiScaleDecoder(channel, nPreLayers, stage)

    def forward(self, x: torch.Tensor, codes=None):
        latents = self._encoder(x)
        if codes is not None:
            quantizeds, logits, negLogPs = self._quantizer(latents, codes)
            return quantizeds, logits, negLogPs
        quantizeds, codes, logits, negLogPs = self._quantizer(latents, codes)
        restored = torch.tanh(self._decoder(quantizeds))
        return restored, codes, latents, negLogPs, logits, quantizeds


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
