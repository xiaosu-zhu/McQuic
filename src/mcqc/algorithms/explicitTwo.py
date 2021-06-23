from typing import Callable, Iterator
from logging import Logger
import math

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from cfmUtils.saver import Saver
from cfmUtils.base import FrequecyHook

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.helpers import evalSSIM, psnr
from mcqc.losses.ssim import MsSSIM
from mcqc.models.whole import Whole
from mcqc import Config

WARMUP_STEP = 20000
def _transformerLR(step):
    step = step + 1
    return min(step / WARMUP_STEP, 0.999999 ** (step - WARMUP_STEP))

INCRE_STEP = 1e8
def _tuneReg(step):
    return step / INCRE_STEP

class ExpTwoStage(Algorithm):
    def __init__(self, config: Config, model: Whole, optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], saver: Saver, savePath:str, continueTrain: bool, logger: Logger):
        super().__init__()
        self._rank = dist.get_rank()
        self._worldSize = dist.get_world_size()
        if self._rank == 0 and saver is None:
            raise AttributeError("Not passing a saver for main process.")
        if self._rank != 0 and saver is not None:
            raise AttributeError("Try passing a saver for sub-process.")
        torch.cuda.set_device(self._rank)

        # if torch.backends.cudnn.version() >= 7603:
        #     self._channelLast = True
        #     model = model.to(memory_format=torch.channels_last)

        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank, broadcast_buffers=False, find_unused_parameters=True)

        if self._rank == 0:
            self._evalSSIM = MsSSIM(size_average=False).to(self._rank)

        # self._optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=Consts.Eps, amsgrad=True)
        self._optimizer = optimizer(config.LearningRate, self._model.parameters(), 1e-5)
        # self._scheduler = scheduler(self._optimizer)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, _transformerLR)

        dist.barrier()

        # self._optimizerD = optimizer(1e-5, self._model.module._discriminator.parameters(), 0)
        # self._schedulerD = scheduler(self._optimizerD)
        self._saver = saver
        self._savePath = savePath
        self._logger = logger
        self._config = config
        self._continue = continueTrain
        if self._rank == 0:
            self._loggingHook = FrequecyHook({100: self._fastHook, 1000: self._mediumHook, 10000: self._slowHook})
        else:
            self._loggingHook = None
        # self._accumulatedBatches = 32 //  config.BatchSize

    @staticmethod
    def _deTrans(image):
        return ((image * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def _fastHook(self, **kwArgs):
        ssimLoss, l1l2Loss, qLoss, reg, step, regCoeff, temp, logits = kwArgs["ssimLoss"], kwArgs["l1l2Loss"], kwArgs["qLoss"], kwArgs["reg"], kwArgs["now"], kwArgs["regCoeff"], kwArgs["temperature"], kwArgs["logits"]
        self._saver.add_scalar("Loss/MS-SSIM", ssimLoss.mean(), global_step=step)
        self._saver.add_scalar("Loss/L1L2", l1l2Loss.mean(), global_step=step)
        self._saver.add_scalar("Loss/QLoss", qLoss.mean(), global_step=step)
        self._saver.add_scalar("Loss/Reg", reg.mean(), global_step=step)
        self._saver.add_scalar("Stat/LR", self._scheduler.get_last_lr()[0], global_step=step)
        self._saver.add_scalar("Stat/Reg", regCoeff, global_step=step)
        self._saver.add_scalar("Stat/Temperature", temp, global_step=step)
        self._saver.add_histogram("Stat/Logit", logits[0], global_step=step)

    def _mediumHook(self, **kwArgs):
        images, restored, testLoader, step, i, temperature, regScale = kwArgs["images"], kwArgs["restored"], kwArgs["testLoader"], kwArgs["now"], kwArgs["i"], kwArgs["temperature"], kwArgs["regScale"]
        self._saver.add_images("Train/Raw", self._deTrans(images), global_step=step)
        self._saver.add_images("Train/Res", self._deTrans(restored), global_step=step)
        uniqueCodes = self._eval(testLoader, step)
        self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=i, temperature=temperature, regScale=regScale)
        self._logger.info("[%3dk]: LR = %.2e, T = %.2e, P = %.2e", (step) // 1000, self._scheduler.get_last_lr()[0], temperature, regScale)
        return uniqueCodes

    def _slowHook(self, **kwArgs):
        step = kwArgs["now"]
        if 100000 <= step <= 130000:
            self._scheduler.step()

    def run(self, trainLoader: torch.utils.data.DataLoader, sampler: torch.utils.data.DistributedSampler, testLoader: torch.utils.data.DataLoader):
        step = 0
        # tristate: None (pure latent), False (quantized with straight-through), True (pure quanitzed)
        e2e = True
        images = None
        regScale = 1.0

        epochSteps = len(trainLoader.dataset) // (self._worldSize * trainLoader.batch_size)

        temperature = 10.0
        initTemp = 10.0
        finalTemp = 0.01
        annealRange = int(1e6 // epochSteps)
        initEpoch = 0

        if self._continue:
            mapLocation = {"cuda:0": f"cuda:{self._rank}"}
            loaded = Saver.load(self._savePath, mapLocation, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=initEpoch, temperature=temperature, regScale=regScale)
            step = loaded["step"]
            temperature = loaded["temperature"]
            initEpoch = loaded["epoch"]
            if self._rank == 0:
                uniqueCodes = self._eval(testLoader, step)
                self._logger.info("Resume training from %3dk step.", step // 1000)
        dist.barrier()

        for i in range(initEpoch, self._config.Epoch):
            sampler.set_epoch(i)
            temperature = initTemp * (finalTemp / initTemp) ** (i / annealRange)
            for images in trainLoader:
                self._model.zero_grad(True)
                images = images.to(self._rank, non_blocking=True)
                (ssimLoss, l1l2Loss, qLoss, reg), (restored, codes, latents, logits, quantizeds) = self._model(images, temperature, e2e)
                (self._config.Coef.ssim * ssimLoss + self._config.Coef.l1l2 * l1l2Loss + self._config.Coef.gen * qLoss + regScale * self._config.Coef.reg * reg).mean().backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()
                self._scheduler.step()
                step += 1
                self._config.Coef.reg = _tuneReg(step)
                # e2e = step % 2 == 0
                if step >= 20000: e2e = None
                if self._loggingHook is not None:
                    with torch.no_grad():
                        results = self._loggingHook(step, ssimLoss=ssimLoss, l1l2Loss=l1l2Loss, qLoss=qLoss, reg=reg, now=step, images=images, restored=restored, testLoader=testLoader, i=i, temperature=temperature, regScale=regScale, regCoeff=self._config.Coef.reg, logits=logits)
                        uniqueCodes = results.get(1000, None)
                        if uniqueCodes is not None:
                            regScale = math.sqrt(self._config.Model.k[0] / uniqueCodes)

    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int):
        if self._logger is None:
            return
        self._model.eval()
        model = self._model.module._compressor
        ssims = list()
        psnrs = list()
        bs = list()
        latents = list()
        qs = list()
        for raw in dataLoader:
            raw = raw.to(self._rank, non_blocking=True)
            latent = model._encoder(raw)
            b = model._quantizer.encode(latent)

            latents.append(latent[0].detach().cpu())
            bs.append(b[0].detach().cpu())

            quantized = model._quantizer.decode(b)

            qs.append(quantized[0].detach().cpu())

            restored = model._decoder(quantized)
            raw = self._deTrans(raw)
            restored = self._deTrans(restored)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(20 * (1.0 / (1.0 - ssim).sqrt()).log10())
            psnrs.append(psnr(restored.detach(), raw.detach()))

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        b = torch.cat(bs, 0).cpu().numpy()
        latent = torch.cat(latents, 0).cpu().numpy()
        qs = torch.cat(qs, 0).cpu().numpy()
        self._logger.info("MS-SSIM: %2.2fdB", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("Eval/Res", restored, global_step=step)
        uniqueCodes = np.unique(b).shape[0]
        self._saver.add_scalar("Eval/UniqueCodes", uniqueCodes, global_step=step)
        # [N, C, H, W] -> mean of [N, H, W]
        self._saver.add_scalar("Eval/QError", ((qs - latent) ** 2).sum(1).mean(), global_step=step)
        np.save("q.npy", qs)
        np.save("z.npy", latent)
        # del bs, zs
        self._model.train()
        return int(uniqueCodes)
