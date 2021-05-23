from math import perm
import torch
from torch import nn
from torch.distributions import Categorical

from mcqc.layers.positional import PositionalEncoding2D


class MaskedLangugeModel(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # contains [START, END, MASK]
        self._specialTokens = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(3, d)))
        self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._random = Categorical(torch.Tensor([0.85, 0.12, 0.015, 0.015]))

        self._ffn = nn.Linear(d, k + 3)

    def _randomMask(self, latent: torch.Tensor, code: torch.Tensor, codebook: torch.Tensor):
        n, h, w = code.shape
        # [raw, MSK, RNG, AS_IS] -> [n, h, w]
        mask = self._random.sample((n, h, w, 1)).byte().to(latent.device)

        # [n, h, w] int of (0 ~ K) for indexing codebook [K, D]
        randomChoice = codebook[torch.randint(0, len(codebook), size=(n, h, w))]

        latent = (mask == 0) * latent.permute(0, 2, 3, 1) + (mask == 2) * randomChoice + (mask == 1) * self._specialTokens[2]

        return latent, mask

    def forward(self, latent, code, codebook):
        # [n, h, w, d], [n, h, w]
        latent, mask = self._randomMask(latent, code, codebook)
        n, h, w, d = latent.shape
        # [h, w, n, d]
        latent = latent.permute(1, 2, 0, 3)
        latent = self._position(latent)

        encoded = self._transformer(latent.reshape(h*w, n, d))

        logit = self._ffn(encoded)
        # [h*w, n, k+2], [h*w, n]
        return logit, mask.reshape(n, -1).permute(1, 0), code.reshape(n, -1).permute(1, 0)
