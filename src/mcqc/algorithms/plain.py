from typing import Callable, Iterator, List
from logging import Logger
import math

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from cfmUtils.saver import Saver
from cfmUtils.base import FrequecyHook
from torch.utils.tensorboard.summary import image

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.helpers import evalSSIM, psnr
from mcqc.losses.ssim import MsSSIM
from mcqc.models.whole import Whole
from mcqc import Config
from mcqc.utils.rANS import Encoder, Decoder

WARMUP_STEP = 20000
def _transformerLR(step):
    step = step + 1
    return min(step / WARMUP_STEP, 0.99999 ** (step - WARMUP_STEP))

INCRE_STEP = 40000
def _tuneReg(step, start=False):
    return 1e-1
    # step = step + 1
    # return 1e-2 * min(step / WARMUP_STEP, 0.9999 ** (step - WARMUP_STEP))

class Plain(Algorithm):
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

        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank, broadcast_buffers=False)

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

        self._imgSize = 512 * 512
        # self._accumulatedBatches = 32 //  config.BatchSize

    @staticmethod
    def _deTrans(image):
        return ((image * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    def _fastHook(self, **kwArgs):
        ssimLoss, l1l2Loss, reg, step, regCoeff, temp, logits = kwArgs["ssimLoss"], kwArgs["l1l2Loss"], kwArgs["reg"], kwArgs["now"], kwArgs["regCoeff"], kwArgs["temperature"], kwArgs["logits"]
        self._saver.add_scalar("Loss/MS-SSIM", ssimLoss.mean(), global_step=step)
        self._saver.add_scalar("Loss/L1L2", l1l2Loss.mean(), global_step=step)
        self._saver.add_scalar("Loss/Reg", reg.mean(), global_step=step)
        self._saver.add_scalar("Stat/LR", self._scheduler.get_last_lr()[0], global_step=step)
        self._saver.add_scalar("Stat/Reg", regCoeff, global_step=step)
        self._saver.add_scalar("Stat/Temperature", temp, global_step=step)
        self._saver.add_histogram("Stat/Logit", logits[0], global_step=step)

    def _mediumHook(self, **kwArgs):
        images, restored, testLoader, step, i, temperature, regScale, maskedImages = kwArgs["images"], kwArgs["restored"], kwArgs["testLoader"], kwArgs["now"], kwArgs["i"], kwArgs["temperature"], kwArgs["regScale"], kwArgs["maskedImages"]
        self._saver.add_images("Train/Raw", self._deTrans(images), global_step=step)
        # self._saver.add_images("Train/Masked", self._deTrans(maskedImages), global_step=step)
        self._saver.add_images("Train/Res", self._deTrans(restored), global_step=step)
        uniqueCodes, ratio = self._eval(testLoader, step)
        self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=i, temperature=temperature, regScale=regScale)
        self._logger.info("[%3dk]: LR = %.2e, T = %.2e, P = %.2e", (step) // 1000, self._scheduler.get_last_lr()[0], temperature, regScale)
        return uniqueCodes, ratio

    def _slowHook(self, **kwArgs):
        step = kwArgs["now"]
        if 100000 <= step <= 130000:
            self._scheduler.step()

    # pylint: disable=too-many-locals,arguments-differ
    def run(self, trainLoader: torch.utils.data.DataLoader, sampler: torch.utils.data.DistributedSampler, testLoader: torch.utils.data.DataLoader):
        step = 0
        # tristate: None (pure latent), False (quantized with straight-through), True (pure quanitzed)
        # uniqueCodes = 2048
        images = None
        regScale = 1.0

        epochSteps = len(trainLoader.dataset) // (self._worldSize * trainLoader.batch_size)

        temperature = 10.0
        initTemp = 10.0
        finalTemp = 1.0
        annealRange = int(40000 // epochSteps)
        initEpoch = 0

        if self._continue:
            mapLocation = {"cuda:0": f"cuda:{self._rank}"}
            loaded = Saver.load(self._savePath, mapLocation, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=initEpoch, temperature=temperature, regScale=regScale)
            step = loaded["step"]
            temperature = loaded["temperature"]
            initEpoch = loaded["epoch"]
            if self._rank == 0:
                self._logger.info("Resume training from %3dk step.", step // 1000)
        if self._rank == 0:
            uniqueCodes, ratio = self._eval(testLoader, step)
            regScale = self._config.Model.k[0] / uniqueCodes.numel()

        for i in range(initEpoch, self._config.Epoch):
            sampler.set_epoch(i)
            temperature = initTemp * (finalTemp / initTemp) ** (i / annealRange)
            # temperature = -1/3 * temperature + 13/3
            for images in trainLoader:
                self._optimizer.zero_grad(True)
                images = images.to(self._rank, non_blocking=True)
                (ssimLoss, l1l2Loss, reg), (restored, codes, maskedImages, logits, quantizeds) = self._model(images, temperature)
                ((self._config.Coef.ssim * ssimLoss + self._config.Coef.l1l2 * l1l2Loss).mean() + regScale * self._config.Coef.reg * reg).backward()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()
                self._scheduler.step()
                step += 1
                self._config.Coef.reg = _tuneReg(step, step > 14000)
                # e2e = step % 2 == 0
                if self._loggingHook is not None:
                    with torch.no_grad():
                        results = self._loggingHook(step, ssimLoss=ssimLoss, l1l2Loss=l1l2Loss, reg=reg, now=step, images=images, maskedImages=maskedImages, restored=restored, testLoader=testLoader, i=i, temperature=temperature, regScale=regScale, regCoeff=self._config.Coef.reg, logits=logits)
                        codesAndCounts = results.get(1000, None)
                        if codesAndCounts is not None:
                            uniqueCodes, ratio = codesAndCounts
                            regScale = self._config.Model.k[0] / uniqueCodes.numel()


    # pylint: disable=protected-access
    @torch.no_grad()
    def _eval(self, dataLoader: torch.utils.data.DataLoader, step: int):
        if self._logger is None:
            return
        self._model.eval()
        model = self._model.module._compressor
        ssims = list()
        psnrs = list()
        bs = [list() for _ in self._config.Model.k]
        latents = list()
        qs = list()
        for raw in dataLoader:
            raw = raw.to(self._rank, non_blocking=True)

            latent = model._encoder(raw)

            # M * [n, c // M, h, w]
            splits = torch.chunk(latent[0], len(model._k), 1)
            lHat = list()
            for i in range(len(model._k)):
                b, _ = model._quantizer[i].encode(splits[i])
                q = model._quantizer[i].decode(b)
                lHat.append(q)
                if not ((0 <= b).all() and (b < 256).all()):
                    raise ValueError("codes overflows 8bit threshold.")
                bs[i].append(b.byte().detach().cpu())
            quantized = torch.cat(lHat, 1)
            restored = torch.tanh(model._decoder([quantized,]))

            latents.append(latent[0].detach().cpu())
            qs.append(quantized.detach().cpu())

            raw = self._deTrans(raw)
            restored = self._deTrans(restored)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(ssim)
            # ssims.append(20 * (1.0 / (1.0 - ssim).sqrt()).log10())
            psnrs.append(psnr(restored.detach(), raw.detach()))

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        bs = [torch.cat(x, 0) for x in bs]
        latent = torch.cat(latents, 0)
        qs = torch.cat(qs, 0)
        self._logger.info("MS-SSIM: %2.2f", ssims.mean())
        self._logger.info("   PSNR: %2.2fdB", psnrs.mean())
        self._saver.add_images("Eval/Res", restored, global_step=step)
        uniqueCodes, counts = torch.unique(bs[0], return_counts=True)
        self._saver.add_scalar("Eval/UniqueCodes", len(uniqueCodes), global_step=step)
        # [N, C, H, W] -> mean of [N, H, W]
        self._saver.add_scalar("Eval/QError", ((qs - latent) ** 2).sum(1).mean(), global_step=step)
        # np.save("q.npy", qs)
        # np.save("z.npy", latent)
        # del bs, zs
        self._model.train()

        encoded, bpp = self._compress(bs)

        self._saver.add_scalar("Eval/BPP", bpp, global_step=step)

        return uniqueCodes.detach().cuda(), (counts / b.numel()).detach().cuda()

    def _compress(self, codes: List[torch.Tensor]):
        compressed = list()
        for i, b in enumerate(codes):
            # list of 256 probs
            prob = self._calculateFreq(b)
            # N * [32, 32]
            for code in b:
                encoder = Encoder()
                for symbol in code.flatten():
                    encoder.encode_symbol(prob.tolist(), int(symbol))
                binary = encoder.get_encoded()
                compressed.append(binary)
        # b: 32 bits per word
        # N * M * [binaries]
        total = 32 * sum(len(b) for b in compressed)
        totalPixel = len(codes[0]) * self._imgSize
        bpp = float(total) / totalPixel
        self._logger.info("%db, BPP: %.4f", total, bpp)
        return compressed, bpp

    def _calculateFreq(self, code: torch.Tensor):
        count = torch.bincount(code.flatten(), minlength=256)
        return count / code.numel()
