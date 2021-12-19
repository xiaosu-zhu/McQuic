from typing import Any

import torch


class MovingMean:
    def __init__(self, momentum=0.9):
        super().__init__()
        self._slots = dict()
        self._alpha = 1 - momentum

    def step(self, key, value: torch.Tensor):
        if key in self._slots:
            mean = self._slots[key]
            mean -= self._alpha * (mean - float(value))
            self._slots[key] = mean
        else:
            mean = value.item()
            self._slots[key] = mean
        return mean

    def __getitem__(self, key: Any) -> float:
        return self._slots[key]
