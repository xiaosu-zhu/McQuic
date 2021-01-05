from typing import Type, Callable, Iterator
from logging import Logger

import torch
from torch import nn
from cfmUtils.saver import Saver
from pytorch_msssim import ms_ssim

from mcqc.algorithms.algorithm import Algorithm


class Plain(Algorithm):
    def __init__(self, model: nn.Module, device: str, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], saver: Saver, logger: Logger, epoch: int):
        super().__init__()
        self._model = model
        if device == "cuda" and torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model.to(device))
        else:
            self._model = self._model.to(device)
        self._device = device
        self._optimizer = optimizer(self._model.parameters())
        self._scheduler = scheduler(self._optimizer)
        self._epoch = epoch
        self._saver = saver
        self._logger = logger

    @staticmethod
    def _deTrans(imaage):
        return ((imaage * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def Run(self, dataLoader: torch.utils.data.DataLoader):
        initTemp = 10.0
        step = 0
        for i in range(self._epoch):
            for j, (images, _) in enumerate(dataLoader):
                images = images.to(self._device, non_blocking=True)
                restored, codes, latents, logitsCompressed = self._model("forward", images, initTemp, j % 2 == 0)
                logitsConsistency = self._model("consistency", logitsCompressed.detach(), initTemp, j % 2 == 0)
                loss = self._loss(images, restored, codes, latents, logitsCompressed, logitsConsistency)
                # if j % 2 == 0:
                #     loss *= 0.1
                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._saver.add_scalar("loss", loss, global_step=step)
                if j % 100 == 0:
                    self._saver.add_image("raw", self._deTrans(images[0]), global_step=step)
                    self._saver.add_image("res", self._deTrans(restored[0]), global_step=step)
                step += 1
            self._scheduler.step()
            initTemp *= 0.9
            self._logger.info("a epoch")

    @staticmethod
    def _loss(images, restored, codes, latents, logitsCompressed, logitsConsistency):
        l2Loss = torch.nn.functional.mse_loss(restored, images)
        # klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
        ssimLoss = 1 - ms_ssim((restored + 1), (images + 1), data_range=2.0)
        return ssimLoss + l2Loss # + 1e-6 * klLoss
