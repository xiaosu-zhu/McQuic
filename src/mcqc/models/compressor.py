from typing import Any

import torch
from torch import nn
from cfmUtils.base import parallelFunction, Module

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import Quantizer


class Compressor(Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder(2048)
        self._quantizer = Quantizer(256, 2048, 0.1)
        self._decoder = Decoder(2048)

        # self._functions.update({
        #     "forward": self._ff,
        #     "consistency": self._consistency,
        #     "loss": self._loss
        # })

    @Module.register("forward")
    def _ff(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        latents = self._encoder(x)
        quantized, codes, logits = self._quantizer("forward", latents)
        restored = self._decoder(quantized)
        return restored, codes, latents, logits

    @Module.register("consistency")
    def _consistency(self, logits: torch.Tensor):
        quantized = self._quantizer("quantize", logits)
        restored = self._decoder(quantized)
        latents = self._encoder(restored)
        _, _, newLogits = self._quantizer("forward", latents)
        return newLogits

    @Module.register("loss")
    def _loss(self, images, restored, codes, latents, logitsCompressed, logitsConsistency):
        l2Loss = ((restored - images) ** 2).sum()
        klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
        return l2Loss + klLoss
