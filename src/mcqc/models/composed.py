import torch
from torch import nn

from mcqc.models.compressor import BaseCompressor


class Composed(nn.Module):
    def __init__(self, compressor: BaseCompressor, criterion: nn.Module):
        super().__init__()
        self._compressor = compressor
        self._criterion = criterion

    def forward(self, x):
        xHat, yHat, stats = self._compressor(x)
        return (xHat, yHat, stats), self._criterion(x, xHat, stats)
