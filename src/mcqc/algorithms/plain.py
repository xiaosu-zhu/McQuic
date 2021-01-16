from typing import Type, Callable, Iterator
from logging import Logger

import torch
from torch import nn
from torch.distributions import Categorical
from cfmUtils.saver import Saver
from cfmUtils.vision.colorSpace import rgb2hsv, hsv2rgb
from pytorch_msssim import ms_ssim

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.helpers import evalSSIM, psnr

def _ssimExp(source, target, datarange):
    return (2.7182818284590452353602874713527 - ms_ssim(source, target, data_range=datarange).exp()) / (1.7182818284590452353602874713527)


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

    def run(self, trainLoader: torch.utils.data.DataLoader, testLoader: torch.utils.data.DataLoader):
        initTemp = 10.0
        step = 0
        for i in range(self._epoch):
            self._model.train()
            for images in trainLoader:
                images = images.to(self._device, non_blocking=True)
                hsvImages = (rgb2hsv((images + 1.) / 2.) - 0.5) / 0.5
                restored, codes, latents, logits, quantizeds, targets = self._model(torch.cat([images, hsvImages], axis=1), initTemp, True)
                loss = self._loss(images, hsvImages, restored, codes, latents, logits, quantizeds, targets)
                self._model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
                self._optimizer.step()
                self._saver.add_scalar("loss", loss, global_step=step)
                if (step + 1) % 100 == 0:
                    self._saver.add_images("soft/raw", self._deTrans(images), global_step=step)
                    self._saver.add_images("soft/res", self._deTrans(restored[:, 3:]), global_step=step)
                if (step + 1) % 1000 == 0:
                    self._eval(testLoader, step)
                    initTemp *= 0.9
                    self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, temp=initTemp)
                    self._logger.info("%3dk steps complete, update: LR = %.2e, T = %.2e", (step + 1) // 1000, self._scheduler.get_last_lr()[0], initTemp)
                if (step + 1) % 10000 == 0:
                    self._scheduler.step()
                step += 1

    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int):
        self._model.eval()
        ssims = list()
        psnrs = list()
        if isinstance(self._model, nn.DataParallel):
            model = self._model.module
        else:
            model = self._model
        model = model.cuda()
        for raw in dataLoader:
            raw = raw.to(self._device, non_blocking=True)
            hsvRaw = (rgb2hsv((raw + 1.) / 2.) - 0.5) / 0.5
            latents = model._encoder(torch.cat([raw, hsvRaw], axis=1))
            b = model._quantizer.encode(latents)
            quantized = model._quantizer.decode(b)
            restored = model._decoder(quantized)
            raw = self._deTrans(raw)
            restored = self._deTrans(restored[:, 3:])
            ssims.append(evalSSIM(restored.detach(), raw.detach(), True))
            psnrs.append(psnr(restored.detach(), raw.detach()))
        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        self._logger.info("MS-SSIM: %2.2fdB", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("eval/res", restored, global_step=step)

    @staticmethod
    def _loss(images, hsvImages, restored, codes, latents, logits, quantizeds, targets):
        hsvR, rgbR = torch.chunk(restored, 2, 1)
        # combined = torch.cat([images, hsvImages], axis=1)
        l2Loss = torch.nn.functional.mse_loss(rgbR, images) + torch.nn.functional.mse_loss(hsvR, hsvImages)
        l1Loss = torch.nn.functional.l1_loss(rgbR, images) + torch.nn.functional.l1_loss(hsvR, hsvImages)
        ssimLoss = 2 - ms_ssim((rgbR + 1), (images + 1), data_range=2.0) - ms_ssim((hsvR + 1), (hsvImages + 1), data_range=2.0)

        # transformerL2 = list()
        # transformerL1 = list()
        # for q, t in zip(quantizeds, targets):
        #     transformerL2.append(torch.nn.functional.mse_loss(q, t))
        #     transformerL1.append(torch.nn.functional.l1_loss(q, t))
        # transformerL2 = sum(transformerL2)
        # transformerL1 = sum(transformerL1)

        # ssimLoss = _ssimExp((rgbR + 1), (images + 1), 2.0) + _ssimExp((hsvR + 1), (hsvImages + 1), 2.0)

        entropies = list()
        for logit in logits:
            # N, K, H, W -> N, H, W, K -> NHW, K
            distributions = Categorical(logit.permute(0, 2, 3, 1).reshape(-1, logit.shape[1]))

            entropies.append(distributions.entropy().mean())

        # klLoss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logitsConsistency, -1), torch.nn.functional.log_softmax(logitsCompressed.detach(), -1), reduction="batchmean", log_target=True)
        # rgbRhsv = (rgb2hsv((rgbR + 1.) / 2.) - 0.5) / 0.5
        # hsvRrgb = (hsv2rgb((hsvR + 1.) / 2.) - 0.5) / 0.5
        # crossL2Loss = torch.nn.functional.mse_loss(hsvRrgb, images) + torch.nn.functional.mse_loss(rgbRhsv, hsvImages)
        # crossSSIM = 2 - ms_ssim((hsvRrgb + 1), (images + 1), data_range=2.0) - ms_ssim((rgbRhsv + 1), (hsvImages + 1), data_range=2.0)
        return l2Loss \
             + l1Loss \
             + ssimLoss \
           # + transformerL1 + transformerL2 \
           # - 1e-3 * sum(entropies) \
           # + 1e-3 * (crossL2Loss + crossSSIM) \
           # + 1e-6 * klLoss \
