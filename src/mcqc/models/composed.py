import torch
from torch import nn

from .compressor import BaseCompressor


class Composed(nn.Module):
    def __init__(self, compressor: BaseCompressor, criterion: nn.Module):
        super().__init__()
        self._compressor = compressor
        self._criterion = criterion

    def forward(self, x):
        xHat, yHat, codes, logits = self._compressor(x)
        loss = self._criterion(x, xHat, codes, logits)
        statistics = {
            "loss": loss,
            "codes": codes,
            "logits": logits
        }
        return xHat, loss, statistics
