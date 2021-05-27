from typing import List

import torch
from torch import nn
from torch.distributions import Categorical

from mcqc.layers.positional import PositionalEncoding2D


class StackedAutoRegressive(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, ks: List[int], rate=0.1):
        super().__init__()
        self._transformer = nn.ModuleList(nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers) for _ in ks[1:])
        self._position = PositionalEncoding2D(d, 120, 120, rate)
        self._ffn = nn.ModuleList(nn.Linear(d, k) for k in ks[1:])

    def predict(self, latents):
        predicts = list()
        for latent, transformer, ffn in zip(latents, self._transformer, self._ffn):
            latent = latent.detach()
            n, d, h, w = latent.shape
            # [h, w, n, d]
            latent = latent.permute(2, 3, 0, 1)
            latent = self._position(latent)
            # [h*w, n, d]
            encoded = transformer(latent.reshape(h*w, n, d))
            # [h*w, n, k]
            logit = ffn(encoded)
            # [n, h*w, k]
            predicts.append(logit.permute(1, 0, 2).argmax(-1).reshape(n, h, w))
        return predicts

    def forward(self, latents, codes):
        logits = list()
        targets = list()
        for latent, code, transformer, ffn in zip(latents, codes, self._transformer, self._ffn):
            latent = latent.detach()
            n, d, h, w = latent.shape
            # [h, w, n, d]
            latent = latent.permute(2, 3, 0, 1)
            latent = self._position(latent)
            # [h*w, n, d]
            encoded = transformer(latent.reshape(h*w, n, d))
            # [h*w, n, k]
            logit = ffn(encoded)
            # [n, h*w, k], [h*w, n]
            logits.append(logit.permute(1, 0, 2))
            targets.append(code.reshape(n, h*w))
        return logits, targets
