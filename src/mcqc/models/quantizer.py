from typing import Callable, List

import torch
from torch import nn
import torch.nn.functional as F
from cfmUtils.base import Module

from mcqc.layers.layerGroup import LayerGroup
from mcqc.layers.gumbelSoftmax import GumbelSoftmax


class _resBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1, activationFn: Callable = None):
        super().__init__()
        self._layers = nn.Sequential(LayerGroup(d, d, rate, activationFn), LayerGroup(d, 2 * d, rate, activationFn), LayerGroup(2 * d, d, rate, activationFn))

    def forward(self, x):
        return x + self._layers(x)


class _incResBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1, activationFn: Callable = None):
        super().__init__()
        self._fc1 = LayerGroup(d, d // 2, rate, activationFn)
        self._fc2 = LayerGroup(d // 2, d // 4, rate, activationFn)
        self._fc3 = LayerGroup(d // 4, d // 2, rate, activationFn)
        self._fc4 = LayerGroup(d // 2, d, rate, activationFn)
        self._res1 = _resBlock(d // 4, rate, activationFn)
        self._res2 = _resBlock(d // 2, rate, activationFn)
        self._res3 = _resBlock(d, rate, activationFn)

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


class Quantizer(nn.Module):
    def __init__(self, k: int, cin: int, rate: float = 0.1):
        super().__init__()
        self._net = nn.Transformer(cin, 8, 1, 1, cin)
        self._net = nn.Linear(cin, k)
        self._codebook = nn.Parameter(torch.randn(k, cin))

    def forward(self, x, temperature, hard):
        x = x.permute(0, 2, 3, 1)
        # [N, h, w, k]
        logits = self._net(x)
        samples = F.gumbel_softmax(logits, temperature, hard)
        # [N, h, w, C] <- [N, h, w, k] @ [k, C]
        quantized = samples @ self._codebook
        return quantized.permute(0, 3, 1, 2), samples, logits.permute(0, 3, 1, 2)


class MultiCodebookQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._net = nn.ModuleList([nn.Linear(cin, numCodewords) for numCodewords in k])
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        samples = list()
        logits = list()
        for x, net, codebook in zip(latents, self._net, self._codebook):
            x = x.permute(0, 2, 3, 1)
            # [N, h, w, k]
            logit = net(x)
            sample = F.gumbel_softmax(logit * self._d, temperature, hard)
            # [N, h, w, C] <- [N, h, w, k] @ [k, C]
            quantized = codebook(sample)

            quantizeds.append(quantized.permute(0, 3, 1, 2))
            samples.append(sample)
            logits.append(logit.permute(0, 3, 1, 2))

        return quantizeds, samples, logits


class TransformerQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._prob = nn.ModuleList([nn.Linear(cin, numCodewords) for numCodewords in k])
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, cin, rate), 6, nn.LayerNorm(cin, 1e-6))
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, cin, rate), 6, nn.LayerNorm(cin, 1e-6))
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        samples = list()
        logits = list()
        targets = list()
        for xRaw, net, codebook, k in zip(latents, self._prob, self._codebook, self._k):
            targets.append(xRaw)
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c] -> [h*w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
            x = self._encoder(encoderIn)
            # [h*w, n, k] -> [n, h*w, k]
            logit = net(x).permute(1, 0, 2)
            sample = F.gumbel_softmax(logit * self._d, temperature, hard)
            # [N, h*w, c] <- [N, h*w, k] @ [k, C]
            quantized = codebook(sample)
            # [n, h*w, c] -> [h*w, n, c]
            quantized = quantized.permute(1, 0, 2)
            mixed = temperature * encoderIn / (temperature + 1) + quantized / (temperature + 1)
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(mixed, quantized).permute(1, 0, 2).reshape(n, h, w, c)
            # [n, c, h, w]
            quantizeds.append(deTransformed.permute(0, 3, 1, 2))
            samples.append(sample)
            logits.append(logit.reshape(n, h, w, k).permute(0, 3, 1, 2))

        return quantizeds, targets, samples, logits

    # @Module.register("quantize")
    # def _quantize(self, logits, temperature, hard):
    #     logits = logits.permute(0, 2, 3, 1)
    #     samples = F.gumbel_softmax(logits, temperature, hard)
    #     # [N, h, w, C] <- [N, h, w, k] @ [k, C]
    #     quantized = samples @ self._codebook
    #     return quantized.permute(0, 3, 1, 2)
