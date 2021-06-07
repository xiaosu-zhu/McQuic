from math import perm
import torch
from torch import nn
from torch.distributions import Categorical

from mcqc.layers.positional import PositionalEncoding2D


class MaskingModel(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._dropout = nn.Dropout(rate, True)
        self._ffn = nn.Linear(d, k)

    def _createInputandMask(self, latent: torch.Tensor):
        n, d, h, w = latent.shape
        latent = latent.permute(2, 3, 0, 1)
        latent = self._position(latent)
        latent = latent.reshape(h*w, n, d)
        # mask should not all True in a row, otherwise output will be nan
        return latent, torch.triu(torch.ones(h*w, h*w, dtype=bool, device=latent.device), diagonal=1)

    def forward(self, latent, code):
        # [hw, n, d], [hw, hw]
        # at i-th row, predict code at (i+1)-th position while x>=(i+1) are masked
        latent, mask = self._createInputandMask(latent.detach())
        hw, n, d = latent.shape
        # [hw, n, d]
        encoded = self._transformer(latent, mask)
        # [hw, n, k]
        # predict logit
        logit = self._ffn(self._dropout(encoded))
        # ignore last row
        # i-th row to predict (i+1)-th code
        # [n, k, hw - 1], [n, hw - 1]
        return logit.permute(1, 2, 0)[:, :, :-1], code.reshape(n, -1)[:, 1:]

    def predict(self, latent, code):
        # [hw, n, d], [hw, hw]
        # predict code at i-th position while x>=i are masked
        latent, mask = self._createInputandMask(latent)
        n, h, w = code.shape
        # [hw, n, d]
        encoded = self._transformer(latent, mask)
        # [hw, n, k]
        # predict logit
        logit = self._ffn(self._dropout(encoded))
        # [hw, n]
        predict = logit.argmax(-1)
        # [n, hw - 1]
        predict = predict.permute(1, 0)[:, :-1]
        code = code.reshape(n, -1)[:, 1:]
        return predict == code
