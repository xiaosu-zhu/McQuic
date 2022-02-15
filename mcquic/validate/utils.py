from typing import Union, List, Tuple

import torch
from torch import nn

class Decibel(nn.Module):
    def __init__(self, upperBound: float) -> None:
        super().__init__()
        self._upperBound = upperBound ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -10 * (x / self._upperBound).log10()


class EMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._shadow: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            self._shadow.copy_(x)
            return self._shadow
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow
