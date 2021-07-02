import torch
import torch
import torch
import torch
from torch import nn


from mcqc.models.encoder import ResidualEncoder
from mcqc.models.decoder import ResidualDecoder
from mcqc.models.quantizer import AttentiveQuantizer


class ParallelQuantizer(nn.Module):
    def __init__(self, m, k, d):
        super().__init__()
        self._quantizers = nn.ModuleList(AttentiveQuantizer(k, d // m, False, False, True) for _ in range(m))

    def forward(self, x):
        pass

class RefModel(nn.Module):
    def __init__(self, m, k, channel):
        super().__init__()
        self._encoder = ResidualEncoder(channel)
        self._decoder = ResidualDecoder(channel)
