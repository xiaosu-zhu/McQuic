import collections
import os
from typing import List, Tuple, Type
from logging import Logger
import math
import copy

from tqdm import tqdm
import torch
from torch.types import Number
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans
from vlutils.saver import Saver
from vlutils.base import FrequecyHook

from mcqc.algorithms.algorithm import Algorithm
from mcqc.datasets.dataset import BasicLMDB
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.evaluation.metrics import MsSSIM, PSNR
from mcqc.models.whole import WholePQBig
from mcqc import Config
from mcqc.utils.training import _ValueTuner
from mcqc.utils.vision import getTrainingFullTransform, getTrainingPreprocess


_logMapping = {
    "distortion": "Loss/Distortion",
    "auxiliary": "Loss/Auxiliary",
    "predict": "Loss/Context",
    "bpp": "Stat/Reg",
    "lr": "Stat/LR",
    "regCoeff": "Stat/Reg",
    "temperature": "Stat/T"
}


class PixelCNN(Algorithm):
    def __init__(self, config: Config, model: WholePQBig, optimizer: Type[torch.optim.Optimizer], scheduler: Type[torch.optim.lr_scheduler._LRScheduler], valueSchedulers: List[Type[_ValueTuner]], saver: Saver, savePath:str, continueTrain: bool, logger: Logger):
        super().__init__()
        self._rank = dist.get_rank()
        self._worldSize = dist.get_world_size()
        if self._rank == 0 and saver is None:
            raise AttributeError("Not passing a saver for main process.")
        if self._rank != 0 and saver is not None:
            raise AttributeError("Try passing a saver for sub-process.")
        torch.cuda.set_device(self._rank)

        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank, broadcast_buffers=False, find_unused_parameters=True)

        if self._rank == 0:
            self._evalSSIM = MsSSIM(sizeAverage=False).to(self._rank)
            self._evalPSNR = PSNR(sizeAverage=False).to(self._rank)

        self._optimizer = optimizer(self._model.parameters(), **config.Optim.params)
        self._optimFn = optimizer
        if scheduler is not None:
            self._scheduler = scheduler(self._optimizer, **config.Schdr.params)
            self._schdrFn = scheduler
        else:
            self._scheduler = None

        self._regScheduler = valueSchedulers[0](**config.RegSchdr.params)
        self._tempScheduler = valueSchedulers[1](**config.TempSchdr.params)

        # dist.barrier(device_ids=[self._rank])

        self._ckpt = config.WarmStart
        self._saver = saver
        self._savePath = savePath
        self._logger = logger
        self._config = config
        self._continue = continueTrain
        if self._rank == 0:
            self._loggingHook = FrequecyHook({self._config.TestFreq: self._slowHook})
        else:
            self._loggingHook = None
        self._best = -1

    @staticmethod
    def _deTrans(image):
        eps = 1e-3
        max_val = 255
        # [-1, 1] to [0, 255]
        return (((image + 1.0) / 2.0).clamp_(0.0, 1.0) * (max_val + 1.0 - eps)).clamp(0.0, 255.0).byte()

    @torch.inference_mode()
    def _trainingStat(self, **kwArgs):
        step = kwArgs["now"]
        self._saver.add_scalar("Stat/Epoch", kwArgs["epoch"], step)
        self._saver.add_scalar("Loss/cls", kwArgs["predict"], global_step=step)
        self._saver.add_scalar("Loss/rate", kwArgs["rate"], global_step=step)
        self._saver.add_scalar(_logMapping["lr"], self._scheduler.get_last_lr()[0], global_step=step)
        self._saver.add_scalar(_logMapping["regCoeff"], self._regScheduler.Value, global_step=step)
        self._saver.add_scalar(_logMapping["temperature"], self._tempScheduler.Value, global_step=step)

    @torch.inference_mode()
    def _slowHook(self, **kwArgs):
        evalLoader, step, epoch = kwArgs["evalLoader"], kwArgs["now"], kwArgs["epoch"]
        ssim, _ = self._eval(evalLoader, step)
        if ssim > self._best:
            self._best = ssim
            path = self._saver._savePath
            self._saver._savePath = os.path.join(self._saver.SaveDir, "best.ckpt")
            self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=epoch, regSchdr=self._regScheduler, tempSchdr=self._tempScheduler)
            self._saver._savePath = path
        self._saver.save(self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=epoch, regSchdr=self._regScheduler, tempSchdr=self._tempScheduler)
        self._logger.info("[%3dk]: LR = %.2e", (step) // 1000, self._scheduler.get_last_lr()[0])

    @torch.inference_mode()
    def _visualizeIntermediate(self, i, code, step):
        # img = latent[0][:, None, ...]
        # fMin, fMax = img.min(), img.max()
        # img = (img - fMin) / (fMax - fMin)
        # img = F.interpolate(img, scale_factor=4, mode="nearest")
        # self._saver.add_images("Train/Feature", img, step)

        # img = prediction[0][:, None, ...].float()
        # img = F.interpolate(img, scale_factor=4, mode="nearest")
        # self._saver.add_images("Train/Predict", img, step)

        code = (code.float() / self._config.Model.k[i] * 255).byte()

        n, m, h, w = code.shape

        code = code.reshape(n * m, 1, h, w)[:32]
        code = F.interpolate(code, scale_factor=4, mode="nearest")
        self._saver.add_images(f"Train/Code{i}", code, step)

    # pylint: disable=too-many-locals,arguments-differ
    def run(self, trainLoader: Prefetcher, sampler: DistributedSampler, evalLoader: DataLoader, testLoader: DataLoader):
        step = 0
        initEpoch = 0
        lastEpoch = 0

        # updateOp = self._model.module._compressor._quantizer.EMAUpdate

        mapLocation = {"cuda:0": f"cuda:{self._rank}"}

        # import copy
        # schdr = copy.deepcopy(self._scheduler)
        if self._continue:
            loaded = Saver.load(self._savePath, mapLocation, True, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=initEpoch, regSchdr=self._regScheduler, tempSchdr=self._tempScheduler)
            step = loaded["step"]
            initEpoch = loaded["epoch"]
            if self._rank == 0:
                self._logger.info("Resume training from %3dk step.", step // 1000)
        elif isinstance(self._ckpt, str) and len(self._ckpt) > 0 and os.path.exists(self._ckpt):
            schdr = copy.deepcopy(self._scheduler)
            regSchdr = copy.deepcopy(self._regScheduler)
            tempSchdr = copy.deepcopy(self._tempScheduler)
            loaded = Saver.load(self._ckpt, mapLocation, False, self._logger, model=self._model, schdr=schdr, step=step, epoch=initEpoch, regSchdr=regSchdr, tempSchdr=tempSchdr)
            step = loaded["step"]
            initEpoch = loaded["epoch"]
            self._optimizer = self._optimFn(self._model.parameters(), **self._config.Optim.params)
            for group in self._optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self._scheduler = self._schdrFn(self._optimizer, last_epoch=schdr.last_epoch, **self._config.Schdr.params)
            self._regScheduler._epoch = regSchdr._epoch
            self._tempScheduler._epoch = tempSchdr._epoch

        # self._scheduler.last_epoch = schdr.last_epoch
        # del schdr
        # self._scheduler.step()

        if self._rank == 0:
            ssim, _ = self._eval(evalLoader, step)
            self._best = ssim
        # self._reSpreadAll()

        totalBatches = len(trainLoader._loader.dataset) // (self._config.BatchSize * self._worldSize) + 1

        for i in range(initEpoch, self._config.Epoch):
            sampler.set_epoch(i + lastEpoch)
            for images in tqdm(trainLoader, ncols=40, bar_format="Epoch [%3d] {n_fmt}/{total_fmt} |{bar}|" % (i + lastEpoch + 1), total=totalBatches, leave=False, disable=self._rank != 0):
                self._optimizer.zero_grad()
                clsLoss, predictRate = self._model(images, self._tempScheduler.Value)
                clsLoss.backward()
                # if True:
                #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                self._optimizer.step()
                step += 1
                # updateOp()
                if step % 2 == 0 and self._loggingHook is not None:
                    self._trainingStat(now=step, epoch=i + lastEpoch + 1, predict=clsLoss, rate=predictRate)
                dist.barrier()
            if self._loggingHook is not None:
                self._loggingHook(i + 1, now=step, images=images, evalLoader=evalLoader, testLoader=testLoader, epoch=i + lastEpoch + 1)
            if self._scheduler is not None:
                self._scheduler.step()
            self._regScheduler.step()
            self._tempScheduler.step()

        if self._rank == 0:
            self._logger.info("Train finished")

    @torch.inference_mode()
    def _eval(self, dataLoader: DataLoader, step: int) -> Tuple[float, float]:
        model = self._model.module
        model.eval()
        ssims = list()
        psnrs = list()
        bs = [[list() for _ in range(self._config.Model.m)] for _ in range(model._levels)]
        totalPixels = 0

        numImages = 0

        # contextPrediction = list()

        countUnique = list()

        for raw in tqdm(dataLoader, ncols=40, bar_format="Val: {n_fmt}/{total_fmt} |{bar}|", leave=False):
            raw = raw.to(self._rank, non_blocking=True)
            n, _, h, w = raw.shape
            totalPixels += n * h * w

            numImages += n

            restored, allCodes = model.test(raw)
            countUnique.append(allCodes[0][:, 0].flatten())

            for i, codesAtLeveli in enumerate(allCodes):
                # enumerate on m, n, h, w
                for m, codeAtPartM in enumerate(codesAtLeveli.permute(1, 0, 2, 3)):
                    # [n, h, w] -> [n, -1] -> extend() n tensors
                    bs[i][m].extend(codeAtPartM.reshape(len(codeAtPartM), -1))

            raw = self._deTrans(raw)
            restored = self._deTrans(restored)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(-10 * (1.0 - ssim).log10())
            psnrs.append(self._evalPSNR(restored.detach(), raw.detach()))

        # contextPrediction = torch.cat(contextPrediction)

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        ssimScore = ssims.mean().item()
        psnrScore = psnrs.mean().item()
        self._logger.info("MS-SSIM: %2.2fdB", ssimScore)
        self._logger.info("   PSNR: %2.2fdB", psnrScore)
        self._saver.add_scalar("Eval/MS-SSIM", ssimScore, global_step=step)
        self._saver.add_scalar("Eval/PSNR", psnrScore, global_step=step)
        self._saver.add_images("Res/Eval", restored, global_step=step)

        # self._logger.info("Context prediction: %.2f%%", contextPrediction.float().mean() * 100.0)


        # img = restored[0][:, None, ...]
        # fMin, fMax = img.min(), img.max()
        # img = (img - fMin) / (fMax - fMin)
        # img = F.interpolate(img, scale_factor=4, mode="nearest")
        # self._saver.add_images("Eval/Feature", img, step)
        # [N, h, w] codes
        self._saver.add_scalar("Eval/UniqueCodes", len(torch.unique(torch.cat(countUnique))), global_step=step)
        # [N, C, H, W] -> mean of [N, H, W]
        # self._saver.add_scalar("Eval/QError", ((qs - zs) ** 2).sum(1).mean(), global_step=step)
        model.train()

        encoded, bpp = self._compress(bs, totalPixels)
        total = 8 * sum(len(binary) for binary in encoded)
        self._logger.info("%.2fMB for %d images, BPP: %.4f", total / 1048576, numImages, bpp)
        self._saver.add_scalar("Eval/BPP", bpp, global_step=step)
        return ssimScore, psnrScore

    @torch.inference_mode()
    def _evalFull(self, dataLoader: DataLoader, step: int) -> Tuple[Number, Number]:
        model = self._model.module._compressor
        ssims = list()
        psnrs = list()
        bs = [[list() for _ in range(self._config.Model.m)] for _ in range(model._levels)]
        totalPixels = 0

        numImages = 0

        # contextPrediction = list()

        countUnique = list()

        minLength = int(32 * (2 ** len(self._config.Model.k)))

        datasetLength = len(dataLoader.dataset)

        for j, raw in enumerate(tqdm(dataLoader, ncols=40, bar_format="Test: {n_fmt}/{total_fmt} |{bar}|", leave=False)):
            raw = raw.to(self._rank, non_blocking=True)
            n, _, h, w = raw.shape

            wCrop = w - math.floor(w / minLength) * minLength
            hCrop = h - math.floor(h / minLength) * minLength
            cropLeft = wCrop // 2
            cropRight = wCrop - cropLeft
            cropTop = hCrop // 2
            cropBottom = hCrop - cropTop

            if cropBottom == 0:
                cropBottom = -h
            if cropRight == 0:
                cropRight = -w

            raw = raw[:, :, cropTop:(-cropBottom), cropLeft:(-cropRight)]

            n, _, h, w = raw.shape

            if h % minLength != 0 or w % minLength != 0:
                raise RuntimeError(f"Cropping not correct, the cropped image is {raw.shape}.")

            totalPixels += n * h * w

            numImages += n

            # -1 in codes can be predicted.
            restored, allCodes = model.test(raw)
            countUnique.append(allCodes[0][:, 0].flatten())

            for i, codesAtLeveli in enumerate(allCodes):
                # enumerate on m, n, h, w
                for m, codeAtPartM in enumerate(codesAtLeveli.permute(1, 0, 2, 3)):
                    # [n, h, w] -> [n, -1] -> extend() n tensors
                    bs[i][m].extend(codeAtPartM.reshape(len(codeAtPartM), -1))

            raw = self._deTrans(raw)

            restored = self._deTrans(restored)

            if (j + 1) % (datasetLength // 5) == 0:
                self._saver.add_images(f"Res/Test_{j // (datasetLength // 5) + 1}", restored, global_step=step)

            ssim = self._evalSSIM(restored.detach().float(), raw.detach().float())

            ssims.append(-10 * (1.0 - ssim).log10())
            psnrs.append(self._evalPSNR(restored.detach(), raw.detach()))

        # contextPrediction = torch.cat(contextPrediction)

        ssims = torch.cat(ssims, 0)
        psnrs = torch.cat(psnrs, 0)
        ssimScore = ssims.mean().item()
        psnrScore = psnrs.mean().item()
        self._logger.info("Test MS-SSIM: %2.2fdB", ssimScore)
        self._logger.info("Test    PSNR: %2.2fdB", psnrScore)
        self._saver.add_scalar("Test/MS-SSIM", ssimScore, global_step=step)
        self._saver.add_scalar("Test/PSNR", psnrScore, global_step=step)

        model.train()

        encoded, bpp = self._compress(bs, totalPixels)
        total = 8 * sum(len(binary) for binary in encoded)
        self._logger.info("%.2fMB for %d images, BPP: %.4f", total / 1048576, numImages, bpp)
        self._saver.add_scalar("Test/BPP", bpp, global_step=step)
        return ssimScore, psnrScore

    def _compress(self, codes: List[List[List[torch.Tensor]]], totalPixels):
        encoder = ans.RansEncoder()
        decoder = ans.RansDecoder()
        compressed = list()
        bits = list()
        allCdfs = list()
        for lv, levels in enumerate(codes):
            images = list()
            cdfs = list()
            for part in levels:
                # N * [-1] all code of images at level i, group m
                c = torch.cat(part)
                pixels = len(c)
                prob = self._calculateFreq(c.flatten(), self._config.Model.k[lv])
                estimateEntropy = prob.log2()
                estimateEntropy[estimateEntropy == float("-inf")] = 0
                estimateEntropy = -(prob * estimateEntropy).sum().item()
                bits.append(estimateEntropy * pixels)
                cdf = pmf_to_quantized_cdf(prob.tolist(), 16)
                cdfs.append(cdf)
                images.append(part)
            # codePerImage: M * [Tensor of [h * w]]
            for codePerImage in zip(*images):
                # [M, h * w]
                codePerImage = torch.stack(codePerImage, 0)
                indices = torch.arange(codePerImage.shape[0])[:, None].expand_as(codePerImage).flatten().int().tolist()
                cdfSizes = [self._config.Model.k[lv] + 2] * self._config.Model.m
                offsets = torch.zeros_like(codePerImage).flatten().int().tolist()
                # params: List of symbols, List of indices of pdfs, List of pdfs, List of upper-bounds, List of offsets
                # [0, 1, 2, 3], [0, 0, 1, 1], [[xx, xx, xx, xx], [xx, xx, xx, xx]], [4, 4, 4, 4], [0, 0, 0, 0]
                binary: str = encoder.encode_with_indexes(codePerImage.flatten().int().tolist(), indices, cdfs, cdfSizes, offsets)
                # [M, h, w] binary
                compressed.append(binary)
                restoredCode = decoder.decode_with_indexes(binary, indices, cdfs, cdfSizes, offsets)
                if torch.any(torch.tensor(restoredCode) != codePerImage.flatten().int().cpu()):
                    raise RuntimeError("Compress error.")
            allCdfs.append(cdfs)
        # binary: 1 byte per word
        # N * [binaries]
        total = 8 * sum(len(binary) for binary in compressed)
        bpp = float(total) / totalPixels

        perfect = sum(bits) / totalPixels
        self._logger.info("Estimate \"perfect\" BPP: %.4f", perfect)
        return compressed, bpp

    def _calculateFreq(self, code: torch.Tensor, k):
        count = torch.bincount(code, minlength=k)
        return count / code.numel()
