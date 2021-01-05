from typing import Any

import torch
from torch import nn
from cfmUtils.base import parallelFunction, Module
from pytorch_msssim import ms_ssim

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import Quantizer


class Compressor(Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder(512)
        self._quantizer = Quantizer(256, 512, 0.1)
        self._decoder = Decoder(512)

    @Module.register("forward")
    def _ff(self, x: torch.Tensor, temperature: float, hard: bool) -> (torch.Tensor, torch.Tensor):
        latents = self._encoder(x)
        quantized, codes, logits = self._quantizer("forward", latents, temperature, hard)
        restored = self._decoder(quantized)
        return restored, codes, latents, logits

    @Module.register("consistency")
    def _consistency(self, logits: torch.Tensor, temperature: float, hard: bool):
        quantized = self._quantizer("quantize", logits, temperature, hard)
        restored = self._decoder(quantized)
        latents = self._encoder(restored)
        _, _, newLogits = self._quantizer("forward", latents, temperature, hard)
        return newLogits

    # @Module.register("loss")
    # def _loss(self, images, restored, codes, latents, logitsCompressed, logitsConsistency):
    #     l2Loss = torch.nn.functional.mse_loss(restored, images)
    #     # klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
    #     ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0)
    #     return ssimLoss + l2Loss # + 1e-6 * klLoss
