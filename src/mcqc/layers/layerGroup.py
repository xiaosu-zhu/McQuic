"""Module of LayerGroup w/ or w/o ReLU version"""
from torch import nn


class LayerGroup(nn.Module):
    def __init__(self, din, dout, rate, relu: bool = True):
        super().__init__()
        seq = [nn.Linear(din, dout), nn.LayerNorm(dout, 1e-6), nn.Dropout(rate, inplace=True)]
        if relu:
            seq.insert(1, nn.LeakyReLU(inplace=True))
        self._seq = nn.Sequential(*seq)

    def forward(self, x):
        return self._seq(x)
