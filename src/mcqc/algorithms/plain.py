from typing import Type, Callable, Iterator
from logging import Logger

import torch
from torch import nn
from cfmUtils.saver import Saver
from cfmUtils.vision import rgb2hsv, hsv2rgb
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
                hsvImages = (rgb2hsv((images + 1.) / 2.) - 0.5) / 0.5
                restored, codes, latents, logitsCompressed, logitsConsistency = self._model(torch.cat([images, hsvImages], axis=1), initTemp, False) # j % 2 == 0)
                loss = self._loss(images, hsvImages, restored, codes, latents, logitsCompressed, logitsConsistency)
                # if j % 2 == 0:
                #     loss *= 0.1
                self._model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
                self._optimizer.step()
                self._saver.add_scalar("loss", loss, global_step=step)
                # if j % 100 == 0:
                #     self._saver.add_images("hard/raw", self._deTrans(images), global_step=step)
                #     self._saver.add_images("hard/res", self._deTrans(restored), global_step=step)
                if j % 100 == 1:
                    self._saver.add_images("soft/raw", self._deTrans(images), global_step=step)
                    self._saver.add_images("soft/res", self._deTrans(restored[:, 3:]), global_step=step)
                step += 1
                if (j + 1) % 1000 == 0:
                    break
            self._scheduler.step()
            initTemp *= 0.9
            self._logger.info("an epoch")

    @staticmethod
    def _loss(images, hsvImages, restored, codes, latents, logitsCompressed, logitsConsistency):
        hsvR, rgbR = torch.chunk(restored, 2, 1)
        # combined = torch.cat([images, hsvImages], axis=1)
        l2Loss = torch.nn.functional.mse_loss(rgbR, images) + torch.nn.functional.mse_loss(hsvR, hsvImages)
        ssimLoss = 2 - ms_ssim((rgbR + 1), (images + 1), data_range=2.0) - ms_ssim((hsvR + 1), (hsvImages + 1), data_range=2.0)
        # klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
        # rgbRhsv = (rgb2hsv((rgbR + 1.) / 2.) - 0.5) / 0.5
        # hsvRrgb = (hsv2rgb((hsvR + 1.) / 2.) - 0.5) / 0.5
        # crossL2Loss = torch.nn.functional.mse_loss(hsvRrgb, images) + torch.nn.functional.mse_loss(rgbRhsv, hsvImages)
        # crossSSIM = 2 - ms_ssim((hsvRrgb + 1), (images + 1), data_range=2.0) - ms_ssim((rgbRhsv + 1), (hsvImages + 1), data_range=2.0)
        return ssimLoss + l2Loss # + 1e-3 * (crossL2Loss + crossSSIM) # + 1e-6 * klLoss
