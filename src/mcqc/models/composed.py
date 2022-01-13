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
        rate, distortion = self._criterion(x, xHat, codes, logits)
        return xHat, (rate, distortion), codes, logits
