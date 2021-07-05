import abc

import torch
from torch import nn


class Test(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError


class Speed:
    def __init__(self, model: nn.Module, device: int) -> None:
        super().__init__()
        self._model = model.to(device)
        self._testInput = torch.rand(8, 3, 1024, 1024).to(device)

    def test(self):
        # warmup
        for _ in range(10):
            self._model(self._testInput)
        torch.cuda.synchronize()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test
        for _ in range(100):
            self._model(self._testInput)
        endEvent.record()
        torch.cuda.synchronize()
        milliSeconds = startEvent.elapsed_time(endEvent)

        return {"averaged_forward_pass": milliSeconds}
