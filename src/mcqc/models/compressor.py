from typing import Any

import torch
from torch import nn
from cfmUtils.base import parallelFunction, Module
from pytorch_msssim import ms_ssim

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import Quantizer


class Compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder(512)
        self._quantizer = Quantizer(256, 512, 0.1)
        self._decoder = Decoder(512)

    def forward(self, x: torch.Tensor, temperature: float, hard: bool):
        latents = self._encoder(x)
        quantized, codes, logits = self._quantizer(latents, temperature, hard)
        restored = self._decoder(quantized)

        # restoredC = self._decoder(quantized.detach())
        # newLatents = self._encoder(restoredC)
        # _, _, newLogits = self._quantizer(newLatents, temperature, hard)

        return restored, codes, latents, logits, None # newLogits
