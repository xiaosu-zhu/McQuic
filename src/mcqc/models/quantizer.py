import torch
from torch import nn
import torch.nn.functional as F
from cfmUtils.base import Module

from mcqc.layers.layerGroup import LayerGroup
from mcqc.layers.gumbelSoftmax import GumbelSoftmax


class _resBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1):
        super().__init__()
        self._layers = nn.Sequential(LayerGroup(d, d, rate), LayerGroup(d, 2 * d, rate), LayerGroup(2 * d, d, rate))

    def forward(self, x):
        return x + self._layers(x)


class _incResBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1):
        super().__init__()
        self._fc1 = LayerGroup(d, d // 2, rate)
        self._fc2 = LayerGroup(d // 2, d // 4, rate)
        self._fc3 = LayerGroup(d // 4, d // 2, rate)
        self._fc4 = LayerGroup(d // 2, d, rate)
        self._res1 = _resBlock(d // 4, rate)
        self._res2 = _resBlock(d // 2, rate)
        self._res3 = _resBlock(d, rate)

    def forward(self, x):
        # d // 2
        x1 = self._fc1(x)
        # d // 4
        x2 = self._fc2(x1)
        # d // 2
        x1 = self._res2(x1)
        # d // 4
        x2 = self._res1(x2)
        # d
        x3 = self._res3(x)
        return x + x3 + self._fc4(self._fc3(x2) + x1)


class Quantizer(Module):
    def __init__(self, k: int, cin: int, rate: float = 0.1):
        super().__init__()
        self._net = nn.Linear(cin, k)
        self._codebook = nn.Parameter(torch.randn(k, cin))

    @Module.register("forward")
    def _ff(self, x, temperature, hard):
        x = x.permute(0, 2, 3, 1)
        # [N, h, w, k]
        logits = self._net(x)
        samples = F.gumbel_softmax(logits, temperature, hard)
        # [N, h, w, C] <- [N, h, w, k] @ [k, C]
        quantized = samples @ self._codebook
        return quantized.permute(0, 3, 1, 2), samples, logits.permute(0, 3, 1, 2)

    @Module.register("quantize")
    def _quantize(self, logits, temperature, hard):
        logits = logits.permute(0, 2, 3, 1)
        samples = F.gumbel_softmax(logits, temperature, hard)
        # [N, h, w, C] <- [N, h, w, k] @ [k, C]
        quantized = samples @ self._codebook
        return quantized.permute(0, 3, 1, 2)
