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

    def forward(self, x: torch.Tensor, weight: torch.Tensor = None):
        n, _, h, w = x.shape
        if self.training:
            if weight is None:
                sample = self._sample.sample((n, 1, h, w)).to(x.device)
            else:
                # [n, h, w] weight -> [n, 1, h, w]
                sample = torch.distributions.Bernoulli(probs=weight).sample()[:, None, ...]
            if self._inplace:
                return x.mul_(sample).mul_(self._scale)
            return x * sample * self._scale
        return x


class GroupDropout(nn.Module):
    def __init__(self, rate, group, inplace=False):
        super().__init__()
        self._preserve = 1 - rate
        self._inplace = inplace
        self._group = group
        self._scale = 1.0 / self._preserve
        self._sample = torch.distributions.Bernoulli(probs=self._preserve)

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        if self.training:
            sample = self._sample.sample((n, self._group, 1, 1, 1)).to(x.device)
            sample = sample.expand(-1, -1, c // self._group, -1, -1).reshape(n, c, 1, 1)
            if self._inplace:
                return x.mul_(sample).mul_(self._scale)
            return x * sample * self._scale
        return x


class AQMasking(nn.Module):
    def __init__(self, rate, inplace=False):
        super().__init__()
        self._preserve = 1 - rate
        self._inplace = inplace
        self._sample = torch.distributions.Bernoulli(probs=self._preserve)

    def forward(self, x: torch.Tensor):
        if self.training:
            m, n, c, h, w = x.shape
            sample = self._sample.sample((m, n, c, 1, 1)).to(x.device)
            # [1, n, c, 1, 1], if all droped, that is a bad sample, discard it
            allDroped = sample.sum(0, keepdim=True) < 1
            allDroped = allDroped.expand_as(sample)
            sample[allDroped] = 1.0
            # calculate scale
            # [1, n, c, 1, 1]
            scale = m / sample.sum(0, keepdim=True)
            if self._inplace:
                return x.mul_(sample).mul_(scale)
            return x * sample * scale
        return x
