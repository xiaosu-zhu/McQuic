import collections
import os
from typing import List, Tuple, Type
from logging import Logger
import math

from sklearn.cluster import MiniBatchKMeans
import torchvision
from tqdm import tqdm
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
from mcqc.datasets.dataset import BasicLMDB
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.evaluation.metrics import MsSSIM, PSNR
from mcqc.models.whole import WholePQ
from mcqc import Config
from mcqc.utils.training import _ValueTuner
from mcqc.utils.vision import getTrainingFullTransform, getTrainingPreprocess


_logMapping = {
    "distortion": "Stat/Distortion",
    "auxiliary": "Stat/Auxiliary",
    "predict": "Stat/Context",
    "bpp": "Stat/Reg",
    "lr": "Stat/LR",
    "regCoeff": "Stat/Reg",
    "temperature": "Stat/T"
}


class New(Algorithm):
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
        self._optimFn = optimizer
        if scheduler is not None:
            self._scheduler = scheduler(self._optimizer, **config.Schdr.params)
            self._schdrFn = scheduler
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
            self._loggingHook = FrequecyHook({2: self._fastHook, self._config.EvalStep: self._mediumHook, self._config.TestStep: self._slowHook, self._config.TestStep * 10: self._testHook})
        else:
            self._loggingHook = None
        self._best = -1

    @staticmethod
    def _deTrans(image):
        return ((image * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()

    @torch.inference_mode()
    def _fastHook(self, **kwArgs):
        step = kwArgs["now"]
        self._saver.add_scalar(_logMapping["distortion"], -10 * kwArgs["distortion"].log10(), global_step=step)
        self._saver.add_scalar(_logMapping["auxiliary"], kwArgs["auxiliary"], global_step=step)
        # self._saver.add_scalar(_logMapping["predict"], kwArgs["predict"], global_step=step)
        # self._saver.add_scalar(_logMapping["bpp"], kwArgs["bpp"], global_step=step)
        self._saver.add_scalar(_logMapping["lr"], self._scheduler.get_last_lr()[0], global_step=step)
        # self._saver.add_scalar(_logMapping["regCoeff"], self._regScheduler.Value, global_step=step)
        # self._saver.add_scalar(_logMapping["temperature"], kwArgs["temperature"], global_step=step)

    @torch.inference_mode()
    def _slowHook(self, **kwArgs):
        evalLoader, step, epoch, temperature = kwArgs["evalLoader"], kwArgs["now"], kwArgs["epoch"], kwArgs["temperature"]
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
    def _testHook(self, **kwArgs):
        testLoader, step = kwArgs["testLoader"], kwArgs["now"]
        self._evalFull(testLoader, step)

    @torch.inference_mode()
    def _mediumHook(self, **kwArgs):
        images, restored, step, logits, codes, step = kwArgs["images"], kwArgs["restored"], kwArgs["now"], kwArgs["logits"], kwArgs["codes"], kwArgs["now"]
        # prediction = kwArgs["prediction"]
        # [n, m, h, w, k]
        # print(logits.shape)
        # self._saver.add_scalar("Stat/MaxProb", logits[0, 0].softmax(-1).max(), global_step=step)
        # self._saver.add_scalar("Stat/MinProb", logits[0, 0].softmax(-1).min(), global_step=step)
        self._saver.add_histogram("Stat/Logit", logits[0, 0], global_step=step)
        for i, c in enumerate(codes):
            self._saver.add_histogram(f"Stat/Code{i}", c[0, ..., 0].flatten(), global_step=step)
            self._visualizeIntermediate(i, c, step)
        self._saver.add_images("Train/Raw", self._deTrans(images), global_step=step)
        self._saver.add_images("Train/Res", self._deTrans(restored), global_step=step)

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
        images = None

        temperature = 1.0
        # finalTemp = 0.001 / math.sqrt(self._config.Model.k[0])
        # annealRate = 0.9
        initEpoch = 0
        lastEpoch = 0

        # updateOp = self._model.module._compressor._quantizer.EMAUpdate

        mapLocation = {"cuda:0": f"cuda:{self._rank}"}

        # import copy
        # schdr = copy.deepcopy(self._scheduler)
        if self._continue:
            loaded = Saver.load(self._savePath, mapLocation, True, self._logger, model=self._model, optim=self._optimizer, schdr=self._scheduler, step=step, epoch=initEpoch, temperature=temperature, regSchdr=self._regScheduler)
            step = loaded["step"]
            initEpoch = loaded["epoch"]
            temperature = loaded["temperature"]
            if self._rank == 0:
                self._logger.info("Resume training from %3dk step.", step // 1000)
        elif isinstance(self._ckpt, str) and len(self._ckpt) > 0 and os.path.exists(self._ckpt):
            loaded = Saver.load(self._ckpt, mapLocation, False, self._logger, model=self._model, epoch=lastEpoch)
            lastEpoch = loaded["epoch"]

        # self._scheduler.last_epoch = schdr.last_epoch
        # del schdr
        # self._scheduler.step()

        if self._rank == 0:
            if self._continue:
                ssim, _ = self._eval(evalLoader, step)
            else:
                ssim, _ = self._evalFull(testLoader, step)
            self._best = ssim
        self._reSpreadAll()

        totalBatches = len(trainLoader._loader.dataset) // (self._config.BatchSize * self._worldSize)

        for i in range(initEpoch, self._config.Epoch):
            if self._saver is not None:
                self._saver.add_scalar("Stat/Epoch", i + lastEpoch, step)

            sampler.set_epoch(i + lastEpoch)
            # ssimCoef = self._config.Coef.ssim / (self._config.Coef.ssim + self._config.Coef.l1l2 + self._regScheduler.Value)
            # contextCoef = self._config.Coef.l1l2 / (self._config.Coef.ssim + self._config.Coef.l1l2 + self._regScheduler.Value)
            # bppCoef = self._regScheduler.Value / (self._config.Coef.ssim + self._config.Coef.l1l2 + self._regScheduler.Value)
            for images in tqdm(trainLoader, ncols=40, bar_format="Epoch [%3d] {n_fmt}/{total_fmt} |{bar}|" % (i + 1), total=totalBatches, leave=False, disable=self._rank != 0):
                self._optimizer.zero_grad()
                dLoss, auxLoss, (restored, allHards, allLogits) = self._model(images, temperature)
                (dLoss + 1e-3 * auxLoss).backward()
                # if True:
                #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                self._optimizer.step()
                step += 1
                # updateOp()
                if self._loggingHook is not None:
                    with torch.inference_mode():
                        self._loggingHook(step, now=step, images=images, restored=restored, evalLoader=evalLoader, testLoader=testLoader, epoch=i, temperature=temperature, logits=allLogits[0], codes=allHards, distortion=dLoss, auxiliary=auxLoss)
                if step % (self._config.TestStep * 10) == 0:
                    self._optimizer.zero_grad()
                    self._reSpreadAll()
                dist.barrier()
            # temperature = max(finalTemp, temperature * annealRate)
            if self._scheduler is not None:
                self._scheduler.step()
            self._regScheduler.step()

    def _reSpreadAll(self):
        if self._rank == 0:
            self._logger.debug("Begin re-assigning...")
            self._reSpread(self._model.module._compressor)
        dist.barrier()
        if self._rank == 0:
            self._logger.debug("Begin broadcast...")
        for i in range(len(self._config.Model.k)):
            for j in range(self._config.Model.m):
                codebook = self._model.module._compressor._quantizers[i][j]._codebook.clone().detach()
                dist.broadcast(codebook, 0)
                self._model.module._compressor._quantizers[i][j]._codebook.data.copy_(codebook)
        del self._optimizer
        # reset optimizer's moments
        self._optimizer = self._optimFn(self._model.parameters(), **self._config.Optim.params)
        # restore learning rate
        lr = self._scheduler.get_last_lr()[0]
        for g in self._optimizer.param_groups:
            g['lr'] = lr
        # replace scheduler's optimizer
        if self._scheduler is not None:
            self._scheduler = self._schdrFn(self._optimizer, **self._config.Schdr.params)
            # self._scheduler.optimizer = self._optimizer
            # self._scheduler.step()
        if self._rank == 0:
            self._logger.debug("End broadcast...")

    @torch.inference_mode()
    def _reSpread(self, model):
        model.eval()
        trainDataset = BasicLMDB(os.path.join("data", self._config.Dataset), maxTxns=(self._config.BatchSize + 4), transform=getTrainingFullTransform())
        dataLoader = DataLoader(trainDataset, batch_size=self._config.BatchSize, shuffle=True, num_workers=self._config.BatchSize + 4, pin_memory=True)
        quantizeds = [[list() for _ in range(self._config.Model.m)] for _ in range(model._levels)]
        for image in tqdm(dataLoader, ncols=40, bar_format="Assign: {n_fmt}/{total_fmt} |{bar}|", leave=False):
            image = image.to(self._rank, non_blocking=True)
            # list of [n, m, h, w]
            allOriginal = model.prepare(image)
            for ori, levelQs in zip(allOriginal, quantizeds):
                # [n, h, w, c]
                for part, partQs in zip(ori.permute(1, 0, 2, 3), levelQs):
                    partQs.append(part.flatten().cpu())

        numNeverAssigned, numAll = 0, 0

        for i, (k, levelQs) in enumerate(zip(self._config.Model.k, quantizeds)):
            for j, partQs in enumerate(levelQs):
                # kmeans = kmeans_core(k, torch.cat(partQs), all_cuda=False, batch_size=3 * k)
                # kmeans.run()
                # codebook = kmeans.cent
                # kmeans = MiniBatchKMeans(n_clusters=k, init="random", compute_labels=False, tol=1e-6)
                # kmeans.fit(torch.cat(partQs).cpu().numpy())
                # codebook = kmeans.cluster_centers_
                codebook = model._quantizers[i][j]._codebook.data
                partQs = torch.cat(partQs)
                # some of the entry is 0
                counts = torch.bincount(partQs, minlength=k)
                neverAssigned = codebook[counts < 1]
                if len(neverAssigned) > k // 2:
                    mask = torch.zeros((len(neverAssigned), ), dtype=counts.dtype, device=counts.device)
                    maskIdx = torch.randperm(len(mask))[k // 2:]
                    mask[maskIdx] = 1
                    counts[counts < 1] = mask
                    neverAssigned = codebook[counts < 1]
                argIdx = torch.argsort(counts, descending=True)[:(k - len(neverAssigned))]
                fullyAssigned = codebook[argIdx]
                selectedIdx = torch.randperm(len(fullyAssigned))[:len(neverAssigned)]
                codebook[counts < 1] = fullyAssigned[selectedIdx]
                self._logger.debug("Re-assign on %d:%d, %d%% are never assigned.", i, j, int(len(neverAssigned) / float(k) * 100))
                numNeverAssigned += len(neverAssigned)
                numAll += k
                # model._quantizers[i][j]._codebook.data.copy_(torch.from_numpy(codebook))
        self._logger.info("Re-assign of %d%% codewords completed.", int(numNeverAssigned / float(numAll) * 100))
        model.train()

    @torch.inference_mode()
    def _eval(self, dataLoader: DataLoader, step: int) -> Tuple[float, float]:
        model = self._model.module._compressor
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
        model.eval()
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
        compressed = list()
        bits = list()
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
        # binary: 1 byte per word
        # N * [binaries]
        total = 8 * sum(len(binary) for binary in compressed)
        bpp = float(total) / totalPixels

        perfect = sum(bits) / totalPixels
        self._logger.info("Estimate \"perfect\" BPP: %.4f", perfect)
        return compressed, bpp

    def _decompressAndCheck(self, rawCodes: List[torch.Tensor], binaries: List[str], cdfs: List[List[float]]):
        decoder = ans32.RansDecoder()
        for binary, raw in zip(binaries, rawCodes):
            m, h, w = raw.shape
            code: List[int] = decoder.decode_with_indexes(binary, torch.arange(m)[:, None, None].expand(m, h, w).flatten().int().tolist(), cdfs, [self._config.Model.k] * m, torch.zeros(m, h, w).flatten().int().tolist())
            code = torch.tensor(code, dtype=torch.long).reshape(m, h, w)
            print(code)
            print(raw)
            input()
            if torch.any(raw != code):
                raise ValueError("Decompress failed, decoded b not equals to raw b.")

    def _calculateFreq(self, code: torch.Tensor, k):
        count = torch.bincount(code, minlength=k)
        return count / code.numel()
