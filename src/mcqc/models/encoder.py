from logging import Logger
import logging

import torch
from torch import nn

from mcqc.layers.convs import conv3x3
from mcqc.layers.blocks import ResidualBlock, ResidualBlockWithStride, AttentionBlock, DownSample
from mcqc.layers.positional import PositionalEncoding2D


class Encoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockWithStride(3, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockWithStride(channel, channel, stride=2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            conv3x3(channel, channel, stride=2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, channel, preProcessNum, scale):
        super().__init__()
        preProcess = [ResidualBlockWithStride(3, channel, stride=2), ResidualBlock(channel, channel)]
        preProcessNum -= 1
        for i in range(preProcessNum):
            preProcess += [ResidualBlockWithStride(channel, channel, stride=2), ResidualBlock(channel, channel)]
        # 1/4, 1/4
        self._preProcess = nn.Sequential(*preProcess)
        # DownSample -> 1/2, 1/2
        self._scales = nn.ModuleList([DownSample(channel) for _ in range(scale)])
        # conv -> 1/2, 1/2
        self._postProcess = nn.ModuleList([conv3x3(channel, channel, stride=1) for _ in range(scale)])

    def forward(self, x: torch.Tensor):
        # 1/4, 1/4
        x = self._preProcess(x)
        results = list()
        for scale, process in zip(self._scales, self._postProcess):
            # each loop: x / 2, result = x / 2 / 2
            x = scale(x)
            results.append(process(x))
        return results

class TransformerEncoder(nn.Module):
    def __init__(self, k:list, cin:int, rate: float = 0.1):
        super().__init__()
        self._encoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, dropout=rate, activation="gelu"), 3)
        self._position = PositionalEncoding2D(cin, 120, 120)
        k = k[0]
        self._finalLayer = nn.Linear(cin, k)
        self._c = cin

    def forward(self, convZs, codebooks):
        latents = list()
        logits = list()
        for xRaw, codebook in zip(convZs, codebooks):
            n, c, h, w = xRaw.shape
            # [k, 1, c]
            codebook = codebook[:, None, :]
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # encoderIn = encoderIn.reshape(-1, n, c)
            # [h*w, n, c] -> [n, c, h, w]
            x = self._encoder(encoderIn, codebook)
            logit = self._finalLayer(x)
            x = x.permute(1, 2, 0).reshape(n, c, h, w)
            logit = logit.permute(1, 0, 2).reshape(n, h, w, -1)
            latents.append(x)
            logits.append(logit)
        return latents, logits
