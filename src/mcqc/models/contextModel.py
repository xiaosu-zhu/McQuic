from math import perm
import torch
from torch import nn
from torch.distributions import Categorical

from mcqc.layers.positional import PositionalEncoding2D


class ContextModel(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # contains [START, END, MASK]
        self._specialTokens = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(3, d)))
        self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._random = Categorical(torch.Tensor([0.85, 0.12, 0.015, 0.015]))

        self._dropout = nn.Dropout(rate, True)
        self._ffn = nn.Linear(d, k + 3)

    def _createInputandMask(self, latent: torch.Tensor):
        n, d, h, w = latent.shape
        latent = latent.permute(2, 3, 0, 1)
        latent = self._position(latent)
        latent = latent.reshape(h*w, n, d)
        return latent, torch.eye(h*w, device=latent.device, dtype=torch.bool)

    def forward(self, latent, code):
        # [hw, n, d], [hw, hw]
        latent, mask = self._createInputandMask(latent)
        hw, n, d = latent.shape
        # [hw, n, d]
        encoded = self._transformer(latent, mask)
        # [hw, n, k]
        logit = self._ffn(self._dropout(encoded))
        # [n, k, hw], [n, hw]
        return logit.permute(1, 2, 0), code.reshape(n, -1)
