from typing import Type, Callable, Iterator
from logging import Logger
import math

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from cfmUtils.saver import Saver
from cfmUtils.vision.colorSpace import rgb2hsv, hsv2rgb
from pytorch_msssim import ms_ssim

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.helpers import evalSSIM, psnr
from mcqc.models.whole import Whole
from mcqc import Consts, Config

def _ssimExp(source, target, datarange):
    return (2.7182818284590452353602874713527 - ms_ssim(source, target, data_range=datarange).exp()) / (1.7182818284590452353602874713527)

WARMUP_RATIO = 10000 ** -1.5

def _transformerLR(epoch):
    epoch = epoch + 1
    return min(epoch ** -0.5, epoch * WARMUP_RATIO)


class TwoStage(Algorithm):
    def __init__(self, config: Config, model: Whole, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], saver: Saver, continueTrain: bool, logger: Logger):
        super().__init__()
        self._rank = dist.get_rank()
        self._worldSize = dist.get_world_size()
        torch.cuda.set_device(self._rank)
        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank)

        # self._optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Consts.Eps, amsgrad=True)
        self._optimizer = optimizer(config.LearningRate, self._model.parameters(), 1e-5)
        self._scheduler = scheduler(self._optimizer)
        # self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, _transformerLR)

        dist.barrier()

        # self._optimizerD = optimizer(1e-5, self._model.module._discriminator.parameters(), 0)
        # self._schedulerD = scheduler(self._optimizerD)
        self._saver = saver
        self._logger = logger
        self._config = config
        self._continue = continueTrain
        # self._accumulatedBatches = 32 //  config.BatchSize

    @staticmethod
    def _deTrans(imaage):
        return ((imaage * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def run(self, trainLoader: torch.utils.data.DataLoader, testLoader: torch.utils.data.DataLoader):
        initTemp = 100.0
        step = 0
        e2e = False
        count = 0
        cv = 1.0
        maxCV = 0.1

        if self._continue:
            loaded = Saver.load(self._saver.SavePath, self._logger, model=self._model)# , optimG=self._optimizerG, schdrG=self._schedulerG, step=step, temp=initTemp)
            # initTemp = loaded["temp"]
            # step = loaded["step"]

        # if self._logger is not None:
        #     self._eval(testLoader, step, e2e)

        for i in range(self._config.Epoch):
            for images in trainLoader:
                self._optimizer.zero_grad()
                images = images.to(self._rank, non_blocking=True)
                (ssimLoss, l1l2Loss, qLoss, reg), (restored, codes, latents, logits, quantizeds) = self._model(images, 1.0, e2e, cv)
                (self._config.Coef.ssim * ssimLoss + self._config.Coef.l1l2 * l1l2Loss + 10 * self._config.Coef.l1l2 * qLoss + self._config.Coef.reg * reg).mean().backward()
                # (self._config.Coef.l1l2 * l1l2Loss + self._config.Coef.reg * reg).mean().backward()
                # (self._config.Coef.ssim * ssimLoss + self._config.Coef.reg * reg).mean().backward()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
                self._optimizer.step()
                if self._saver is not None:
                    with torch.no_grad():
                        # cv = 10 ** float(l1l2Loss.mean() * -10)
                        # gatherOp = dist.gather(ssimLoss, async_op=True)
                        if (step + 1) % 100 == 0:
                            self._saver.add_scalar("loss/ssimLoss", ssimLoss.mean(), global_step=step)
                            self._saver.add_scalar("loss/l1l2Loss", l1l2Loss.mean(), global_step=step)
                            self._saver.add_scalar("loss/qLoss", qLoss.mean(), global_step=step)
                            self._saver.add_scalar("loss/reg", reg.mean(), global_step=step)
                        if (step + 1) % 1000 == 0:
                            self._saver.add_images("train/raw", self._deTrans(images), global_step=step)
                            self._saver.add_images("train/res", self._deTrans(restored), global_step=step)
                            # initTemp = min(initTemp * 1.1, 1.0)
                            self._eval(testLoader, step, e2e)
                            # if dB > target and not flag:
                            #     flag = True
                            #     self._logger.info("Insert Transformer")
                            #     del self._optimizer
                            #     del self._scheduler
                            #     self._createFn()
                            self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step+1, temp=initTemp)
                            self._logger.info("%3dk steps complete, update: LR = %.2e, T = %.2e, count = %d", (step + 1) // 1000, self._scheduler.get_last_lr()[0], initTemp, count)
                if (step + 1) % 10000 == 0 and 100000 < step < 130000:
                    e2e = True
                    # self._schedulerD.step()
                    self._scheduler.step()
                # initTemp = max(initTemp * 0.9999, minTemp)
                step += 1
                # cv *= min(cv * 1.0001, maxCV)
                # mixin *= 0.9999


    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int, transform: bool):
        self._model.eval()
        ssims = list()
        psnrs = list()
        model = self._model.module._compressor
        bs = list()
        for raw in dataLoader:
            raw = raw.to(self._rank, non_blocking=True)

            # restored, _, _, _, _ = self._model(raw, 0.5, True, 0.0)
            latents = model._encoder(raw)
            b, z = model._quantizer.encode(latents, transform)
            bs.append(b[0].detach().cpu())

            quantized = model._quantizer.decode(b, transform)
            restored = model._decoder(quantized)
            raw = self._deTrans(raw)
            restored = self._deTrans(restored)
            ssims.append(evalSSIM(restored.detach(), raw.detach(), True))
            psnrs.append(psnr(restored.detach(), raw.detach()))
        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        b = torch.cat(bs, 0).cpu().numpy()
        np.save("b.npy", b)
        # np.save("c.npy", self._model.module.codebook.weight.detach().cpu().numpy())
        # np.save("z.npy", torch.cat(zs, 0).cpu().numpy())
        # exit()
        self._logger.info("MS-SSIM: %2.2fdB", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("eval/res", restored, global_step=step)
        self._saver.add_scalar("eval/unique_codes", np.unique(b).shape[0], global_step=step)
        del bs, zs
        self._model.train()
        return float(psnrs.mean())
