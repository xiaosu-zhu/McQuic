import torch
from torch import nn


class ChannelWiseDropout(nn.Module):
    def __init__(self, rate, inplace=False):
        super().__init__()
        self._preserve = 1 - rate
        self._inplace = inplace
        self._scale = 1.0 / self._preserve
        self._sample = torch.distributions.Bernoulli(probs=self._preserve)

    def forward(self, x: torch.Tensor):
        n, c, _, _ = x.shape
        if self.training:
            sample = self._sample.sample((n, c, 1, 1)).to(x.device)
            if self._inplace:
                return x.mul_(sample).mul_(self._scale)
            return x * sample * self._scale
        return x


class PointwiseDropout(nn.Module):
    def __init__(self, rate, inplace=False):
        super().__init__()
        self._preserve = 1 - rate
        self._inplace = inplace
        self._scale = 1.0 / self._preserve
        self._sample = torch.distributions.Bernoulli(probs=self._preserve)

    def forward(self, x: torch.Tensor):
        n, _, h, w = x.shape
        if self.training:
            sample = self._sample.sample((n, 1, h, w)).to(x.device)
            if self._inplace:
                return x.mul_(sample).mul_(self._scale)
            return x * sample * self._scale
        return x
