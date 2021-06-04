from math import perm
import torch
from torch import nn
from torch.distributions import Categorical

from mcqc.layers.positional import PositionalEncoding2D


class MaskingModel(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # contains [START, END, MASK]
        self._specialTokens = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(3, d)))
        self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._dropout = nn.Dropout(rate, True)
        self._ffn = nn.Linear(d, k)

    def _createInputandMask(self, latent: torch.Tensor):
        n, d, h, w = latent.shape
        latent = latent.permute(2, 3, 0, 1)
        latent = self._position(latent)
        latent = latent.reshape(h*w, n, d)
        return latent, torch.triu(torch.ones(h*w, h*w, dtype=bool, device=latent.device))

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

    def predict(self, latent, code):
        n, d, h, w = latent.shape
        latent = latent.permute(2, 3, 0, 1)
        latent = self._position(latent)
        latent = latent.reshape(h*w, n, d)
        predicts = list()
        for i in range(h*w):
            # [?, n, d] -> [?, n, d] -> pick the last -> [n, d]
            encoded = self._transformer(latent[:(i+1)])[-1]
            # [n, k]
            logit = self._ffn(encoded)
            # [n] at position i
            predict = logit.argmax(-1)
            predicts.append(predict)
        # [n, h*w] -> [n, h, w]
        predicts = torch.stack(predicts, -1).reshape((n, h, w))
        return predicts == code
