import os
from typing import List, Tuple, Type
from logging import Logger

import torch
from torch.types import Number
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans
from cfmUtils.saver import Saver
from cfmUtils.base import FrequecyHook

from mcqc.algorithms.algorithm import Algorithm
from mcqc.evaluation.metrics import MsSSIM, PSNR
from mcqc.models.whole import WholePQ
from mcqc import Config
from mcqc.utils.training import _ValueTuner


class Plain(Algorithm):
    def __init__(self, config: Config, model: WholePQ, optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], regScheduler: Type[_ValueTuner], saver: Saver, savePath:str, continueTrain: bool, logger: Logger):
        super().__init__()
        self._rank = dist.get_rank()
        self._worldSize = dist.get_world_size()
        if self._rank == 0 and saver is None:
            raise AttributeError("Not passing a saver for main process.")
        if self._rank != 0 and saver is not None:
            raise AttributeError("Try passing a saver for sub-process.")
        torch.cuda.set_device(self._rank)

        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank, broadcast_buffers=False)

        if self._rank == 0:
            self._evalSSIM = MsSSIM(sizeAverage=False).to(self._rank)
            self._evalPSNR = PSNR(sizeAverage=False).to(self._rank)

        self._optimizer = optimizer(self._model.parameters(), **config.Optim.params)
        if scheduler is not None:
            self._scheduler = scheduler(self._optimizer, **config.Schdr.params)
        else:
            self._scheduler = None

        self._regScheduler = regScheduler(**config.RegSchdr.params)

        # dist.barrier(device_ids=[self._rank])

        self._ckpt = config.WarmStart
        self._saver = saver
        self._savePath = savePath
        self._logger = logger
        self._config = config
        self._continue = continueTrain
        if self._rank == 0:
            self._loggingHook = FrequecyHook({500: self._fastHook, self._config.EvalStep: self._mediumHook})
        else:
            self._loggingHook = None
        self._best = -1

    @staticmethod
    def _deTrans(image):
        return ((image * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    @torch.inference_mode()
    def _fastHook(self, **kwArgs):
        images, restored, step, logits, quantized, codes = kwArgs["images"], kwArgs["restored"], kwArgs["now"], kwArgs["logits"], kwArgs["quantized"], kwArgs["codes"]
        # self._saver.add_scalar("Loss/MS-SSIM", -10 * ssimLoss.mean(), global_step=step)
        self._saver.add_histogram("Stat/Logit", logits[0], global_step=step)
        self._visualizeIntermediate(quantized, codes, step)
        self._saver.add_images("Train/Raw", self._deTrans(images), global_step=step)
        self._saver.add_images("Train/Res", self._deTrans(restored), global_step=step)

    @torch.inference_mode()
    def _slowHook(self, **kwArgs):
        testLoader, step = kwArgs["testLoader"], kwArgs["now"]
        ssim, psnr = self._evalFull(testLoader, step)

    @torch.inference_mode()
    def _mediumHook(self, **kwArgs):
        evalLoader, step, epoch, temperature = kwArgs["evalLoader"], kwArgs["now"], kwArgs["epoch"], kwArgs["temperature"]
        if step % self._config.TestStep == 0:
            return
        ssim, _ = self._eval(evalLoader, step)
        if ssim > self._best:
            self._best = ssim
            path = self._saver._savePath
            self._saver._savePath = os.path.join(self._saver.SaveDir, "best.ckpt")
            self._saver.save(self._logger, model=self._model, step=step, epoch=epoch)
            self._saver._savePath = path
        self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=epoch, temperature=temperature, regSchdr=self._regScheduler)
        self._logger.info("[%3dk]: LR = %.2e, T = %.2e", (step) // 1000, self._scheduler.get_last_lr()[0], temperature)

    @torch.inference_mode()
    def _visualizeIntermediate(self, latent, code, step):
        img = latent[0][:, None, ...]
        fMin, fMax = img.min(), img.max()
        img = (img - fMin) / (fMax - fMin)
        img = F.interpolate(img, scale_factor=4, mode="nearest")
        self._saver.add_images("Train/Feature", img, step)

        code = code.byte()

        n, m, h, w = code.shape

        code = code.reshape(n * m, 1, h, w)[:32]
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        self._saver.add_images("Train/Code", code, step)

    # pylint: disable=too-many-locals,arguments-differ
    def run(self, trainLoader: DataLoader, sampler: DistributedSampler, evalLoader: DataLoader, testLoader: DataLoader):
        step = 0
        # tristate: None (pure latent), False (quantized with straight-through), True (pure quanitzed)
        # uniqueCodes = 2048
        images = None

        temperature = 1.0
        finalTemp = 0.001
        annealRate = 0.9999
        initEpoch = 0
        lastEpoch = 0

        updateOps = [q.EMAUpdate for q in self._model.module._compressor._quantizer]

        mapLocation = {"cuda:0": f"cuda:{self._rank}"}

        if self._continue:
            loaded = Saver.load(self._savePath, mapLocation, True, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=initEpoch, temperature=temperature) #, regSchdr=self._regScheduler)
            step = loaded["step"]
            initEpoch = loaded["epoch"]
            temperature = loaded["temperature"]
            if self._rank == 0:
                self._logger.info("Resume training from %3dk step.", step // 1000)
        elif isinstance(self._ckpt, str) and len(self._ckpt) > 0 and os.path.exists(self._ckpt):
            loaded = Saver.load(self._ckpt, mapLocation, False, self._logger, model=self._model, epoch=lastEpoch)
            lastEpoch = loaded["epoch"]
        if self._rank == 0:
            ssim, _ = self._eval(evalLoader, step)
            self._best = ssim

        for i in range(initEpoch, self._config.Epoch):
            sampler.set_epoch(i + lastEpoch)
            for images in trainLoader:
                self._optimizer.zero_grad(True)
                (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, frequencyMaps) = self._model(images, temperature)
                ((self._config.Coef.ssim * ssimLoss + self._config.Coef.l1l2 * l1l2Loss).mean() + self._regScheduler.Value * reg).backward()
                # if True:
                #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                self._optimizer.step()
                step += 1
                for op in updateOps:
                    op()
                if self._loggingHook is not None:
                    with torch.inference_mode():
                        ssim = ssimLoss
                        self._saver.add_scalar("Loss/MS-SSIM", -10 * ssim.log10(), global_step=step)
                        self._saver.add_scalar("Loss/L1L2", l1l2Loss, global_step=step)
                        self._saver.add_scalar("Loss/Reg", reg, global_step=step)
                        self._saver.add_scalar("Stat/LR", self._scheduler.get_last_lr()[0], global_step=step)
                        self._saver.add_scalar("Stat/Reg", self._regScheduler.Value, global_step=step)
                        self._saver.add_scalar("Stat/Temperature", temperature, global_step=step)
                        self._saver.add_scalar("Stat/Freq", (frequencyMaps[0] > 4).sum() / len(frequencyMaps[0]), global_step=step)
                        self._loggingHook(step, now=step, images=images, restored=restored, evalLoader=evalLoader, testLoader=testLoader, epoch=i, temperature=temperature, logits=logits, quantized=quantized, codes=codes)
            temperature = max(finalTemp, temperature * annealRate)
            if self._scheduler is not None:
                self._scheduler.step()
            self._regScheduler.step()

    @torch.inference_mode()
    def _eval(self, dataLoader: DataLoader, step: int) -> Tuple[float, float]:
        self._model.eval()
        model = self._model.module._compressor
        ssims = list()
        psnrs = list()
        bs = [list() for _ in range(self._config.Model.m)]
        zs = list()
        qs = list()
        totalPixels = 0
        for raw in dataLoader:
            raw = raw.to(self._rank, non_blocking=True)
            n, _, h, w = raw.shape
            totalPixels += n * h * w

            latent = model._encoder(raw)
            # M * [n, c // M, h, w]
            splits = torch.chunk(latent, self._config.Model.m, 1)
            lHat = list()
            for i in range(self._config.Model.m):
                b = model._quantizer[i].encode(splits[i])
                q = model._quantizer[i].decode(b)
                lHat.append(q)
                bs[i].extend(x[None, ...] for x in b.int().detach().cpu())
            quantized = torch.cat(lHat, 1)
            # b = model._quantizer.encode(latent)
            # quantized = model._quantizer.decode(b)
            # for i, bb in enumerate(bs):
            #     bb.extend(x[None, ...] for x in b[i].int().detach().cpu())
            restored = torch.tanh(model._decoder(quantized))

            # restored = restored[:, :, :h, :w]

            zs.append(latent.detach().cpu())
            qs.append(quantized.detach().cpu())

            raw = self._deTrans(raw)
            restored = self._deTrans(restored)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(-10 * (1.0 - ssim).log10())
            psnrs.append(self._evalPSNR(restored.detach(), raw.detach()))

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        zs = torch.cat(zs, 0)
        qs = torch.cat(qs, 0)
        ssimScore = ssims.mean().item()
        psnrScore = psnrs.mean().item()
        self._logger.info("MS-SSIM: %2.2fdB", ssimScore)
        self._logger.info("   PSNR: %2.2fdB", psnrScore)
        self._saver.add_scalar("Eval/MS-SSIM", ssimScore, global_step=step)
        self._saver.add_scalar("Eval/PSNR", psnrScore, global_step=step)
        self._saver.add_images("Eval/Res", restored, global_step=step)
        n, h, w = b.shape
        code = b.reshape(n, 1, h, w).byte()
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        self._saver.add_images("Eval/Code", code, step)
        img = quantized[0][:, None, ...]
        fMin, fMax = img.min(), img.max()
        img = (img - fMin) / (fMax - fMin)
        img = F.interpolate(img, scale_factor=4, mode="nearest")
        self._saver.add_images("Eval/Feature", img, step)
        uniqueCodes, _ = torch.unique(torch.cat(bs[0]), return_counts=True)
        self._saver.add_scalar("Eval/UniqueCodes", len(uniqueCodes), global_step=step)
        # [N, C, H, W] -> mean of [N, H, W]
        # self._saver.add_scalar("Eval/QError", ((qs - zs) ** 2).sum(1).mean(), global_step=step)
        self._model.train()

        encoded, bpp = self._compress(bs, totalPixels)
        self._saver.add_scalar("Eval/BPP", bpp, global_step=step)

        return ssimScore, psnrScore

    @torch.inference_mode()
    def _evalFull(self, dataLoader: DataLoader, step: int) -> Tuple[Number, Number]:
        self._model.eval()
        model = self._model.module._compressor
        ssims = list()
        psnrs = list()
        bs = [list() for _ in range(self._config.Model.m)]
        totalPixels = 0
        for k, raw in enumerate(dataLoader):
            raw = raw.to(self._rank, non_blocking=True)
            n, _, h, w = raw.shape
            totalPixels += n * h * w

            latent = model._encoder(raw)
            # M * [n, c // M, h, w]
            splits = torch.chunk(latent, self._config.Model.m, 1)
            lHat = list()
            for i in range(self._config.Model.m):
                b = model._quantizer[i].encode(splits[i])
                q = model._quantizer[i].decode(b)
                lHat.append(q)
                bs[i].append(b.int().detach().cpu())
            quantized = torch.cat(lHat, 1)
            restored = torch.tanh(model._decoder(quantized))

            restored = restored[:, :, :h, :w]

            raw = self._deTrans(raw)
            restored = self._deTrans(restored)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(-10 * (1.0 - ssim).log10())
            if k < 10:
                self._saver.add_image(f"Test/{k}", restored[0], step)
            psnrs.append(self._evalPSNR(restored.detach(), raw.detach()))

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        ssimScore = ssims.mean().item()
        psnrScore = psnrs.mean().item()
        self._logger.info("Test: MS-SSIM: %2.2fdB", ssimScore)
        self._logger.info("         PSNR: %2.2fdB", psnrScore)
        self._saver.add_scalar("Eval/MS-SSIM", ssimScore, global_step=step)
        self._saver.add_scalar("Eval/PSNR", psnrScore, global_step=step)
        self._model.train()
        encoded, bpp = self._compress(bs, totalPixels)
        self._saver.add_scalar("Eval/BPP", bpp, global_step=step)
        return ssimScore, psnrScore

    def _compress(self, codes: List[List[torch.Tensor]], totalPixels):
        compressed = list()
        cdfs = list()
        # b: Tensor of [N, 32, 32]
        for b in codes:
            b = [x.flatten() for x in b]
            # list of 256 probs
            prob = self._calculateFreq(torch.cat(b), self._config.Model.k)
            cdf = pmf_to_quantized_cdf(prob.tolist(), 16)
            # M * [cdf]
            cdfs.append(cdf)
        encoder = ans.RansEncoder()
        # codePerImage: M * [Tensor of [h * w]]
        for codePerImage in zip(*codes):
            # [M, h, w]
            codePerImage = torch.cat(codePerImage, 0)
            # params: List of symbols, List of indices of pdfs, List of pdfs, List of upper-bounds, List of offsets
            # [0, 1, 2, 3], [0, 0, 1, 1], [[xx, xx, xx, xx], [xx, xx, xx, xx]], [4, 4, 4, 4], [0, 0, 0, 0]
            binary = encoder.encode_with_indexes(codePerImage.flatten().int().tolist(), torch.arange(codePerImage.shape[0])[:, None, None].expand_as(codePerImage).flatten().int().tolist(), cdfs, [self._config.Model.k] * self._config.Model.m, torch.zeros_like(codePerImage).flatten().int().tolist())
            compressed.append(binary)
        # binary: 1 byte per word
        # N * [binaries]
        total = 8 * sum(len(binary) for binary in compressed)
        bpp = float(total) / totalPixels
        self._logger.info("%.2fMB for %d images, BPP: %.4f", total / 1048576, len(codes[0]), bpp)
        return compressed, bpp

    def _calculateFreq(self, code: torch.Tensor, k):
        count = torch.bincount(code, minlength=k)
        return count / code.numel()
