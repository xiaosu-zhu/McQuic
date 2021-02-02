from typing import Any
import sys

import torch
from torch import nn
import torch.nn.functional as F
from cfmUtils.base import parallelFunction, Module
from pytorch_msssim import ms_ssim

from .encoder import Encoder, MultiScaleEncoder
from .decoder import Decoder, MultiScaleDecoder
from .quantizer import Quantizer, MultiCodebookQuantizer, TransformerQuantizer, VQuantizer


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder(512)
        self._quantizer = Quantizer(2048, 512, 0.1)
        self._decoder = Decoder(512)

    def forward(self, x: torch.Tensor, temperature: float, hard: bool):
        latents = self._encoder(x)
        quantized, codes, logits = self._quantizer(latents, temperature, hard)
        restored = self._decoder(quantized)

        # restoredC = self._decoder(quantized.detach())
        # newLatents = self._encoder(restoredC)
        # _, _, newLogits = self._quantizer(newLatents, temperature, hard)

        return restored, codes, latents, logits, None # newLogits


class MultiScaleCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = MultiScaleEncoder(512, 1)
        self._quantizer = TransformerQuantizer([256], 512, 0.1)
        self._decoder = MultiScaleDecoder(512, 1)

    def forward(self, x: torch.Tensor, temperature: float, hard: bool, e2e: bool):
        latents = self._encoder(x)
        quantizeds, codes, logits = self._quantizer(latents, temperature, hard)

        # if mix:
        #     mixeds = list()
        #     for latent, quantized in zip(latents, quantizeds):
        #         mixed = (mixin * latent.detach() / (mixin + 1)) + (quantized / (mixin + 1))
        #         mixeds.append(mixed)
        #     restored = self._decoder(mixeds)
        # else:
        if e2e:
            restored = self._decoder(quantizeds)
        else:
            restored = self._decoder(latents)
        # restoredC = self._decoder(quantized.detach())
        # newLatents = self._encoder(restoredC)
        # _, _, newLogits = self._quantizer(newLatents, temperature, hard)

        return restored, codes, latents, logits, quantizeds # newLogits
