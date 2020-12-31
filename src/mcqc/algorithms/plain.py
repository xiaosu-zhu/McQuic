from typing import Type, Callable, Iterator

import torch
from torch import nn

from .algorithm import Algorithm


class Plain(Algorithm):
    def __init__(self, model: nn.Module, device: str, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], epoch: int):
        super().__init__()
        self._model = model
        if device == "cuda" and torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model)
        self._model = self._model.to(device)
        self._device = device
        self._optimizer = optimizer(self._model.parameters())
        self._scheduler = scheduler(self._optimizer)
        self._epoch = epoch

    def Run(self, dataLoader: torch.utils.data.DataLoader):
        for _ in range(self._epoch):
            for images, _ in dataLoader:
                images = images.to(self._device, non_blocking=True)
                restored, codes, latents, logitsCompressed = self._model("forward", images)
                logitsConsistency = self._model("consistency", logitsCompressed.detach())
                loss = self._model("loss", images, restored, codes, latents, logitsCompressed, logitsConsistency)
                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()
            self._scheduler.step()
