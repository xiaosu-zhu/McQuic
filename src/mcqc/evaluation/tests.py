import abc
import os
from typing import Dict, List
import torchvision
import shutil

from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mcqc.config import Config
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans

from mcqc.evaluation.metrics import MsSSIM, psnr


def deTrans(image):
    return (image * 255).clamp(0.0, 255.0)


class Test(abc.ABC):
    def __init__(self, config: Config, encoder: nn.Module, decoder: nn.Module, preProcess: nn.Module, postProcess: nn.Module, **kwArgs):
        super().__init__()
        self._config = config
        self._encoder = encoder
        self._decoder = decoder
        self._preProcess = preProcess
        self._postProcess = postProcess
        self._encoder.eval()
        self._decoder.eval()
        self._preProcess.eval()
        self._postProcess.eval()
        torch.autograd.set_grad_enabled(False)
        setattr(self, "test", torch.inference_mode()(self.test))

    @abc.abstractmethod
    def test(self) -> Dict[str, float]:
        raise NotImplementedError


class Speed(Test):
    def __init__(self, **kwArgs) -> None:
        super().__init__(**kwArgs)
        # same as kodak
        self._testInput = torch.rand(24, 3, 768, 512).cuda()

    def test(self):
        x, cAndPadding = self._preProcess(self._testInput)
        # warmup
        for _ in trange(10):
            b, cAndPadding = self._encoder(x, cAndPadding)
            self._decoder(b, cAndPadding)
        torch.cuda.synchronize()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)


        startEvent.record()
        # test encoder
        for _ in trange(100):
            self._encoder(x, cAndPadding)
        endEvent.record()
        torch.cuda.synchronize()
        encoderMs = startEvent.elapsed_time(endEvent) / (100 * len(self._testInput))

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test encoder
        for _ in trange(100):
            self._decoder(b, cAndPadding)
        endEvent.record()
        torch.cuda.synchronize()
        decoderMs = startEvent.elapsed_time(endEvent) / (100 * len(self._testInput))

        return {"encoderForwardTime": encoderMs, "decoderForwardTime": decoderMs}


class Performance(Test):
    def __init__(self, dataset: Dataset, **kwArgs):
        super().__init__(**kwArgs)
        self._dataLoader = DataLoader(dataset, pin_memory=True)
        self._ssim = MsSSIM(sizeAverage=False).cuda()

    def test(self):
        shutil.rmtree("ckpt/images", ignore_errors=True)
        os.makedirs("ckpt/images", exist_ok=True)
        ssims = list()
        psnrs = list()
        bs = [[list() for _ in range(self._config.Model.m)] for _ in range(len(self._config.Model.k))]
        pixels = list()
        for i, x in enumerate(tqdm(self._dataLoader)):
            x = x.cuda(non_blocking=True)
            _, _, h, w = x.shape
            xPadded, cAndPadding = self._preProcess(x)
            # list of [1, ?, ?, m]
            b, cAndPadding = self._encoder(xPadded, cAndPadding)
            y, cAndPadding = self._decoder(b, cAndPadding)
            y = self._postProcess(y, cAndPadding)
            x, y = deTrans(x), deTrans(y)

            ssims.append(float(-10 * (1.0 - self._ssim(x, y)).log10()))
            psnrs.append(float(psnr(x, y)))
            # list of [m, ?]
            for i, codesAtLeveli in enumerate(b):
                # enumerate on m, n, h, w <- n, h, w, m
                for m, codeAtPartM in enumerate(codesAtLeveli.permute(3, 0, 1, 2)):
                    # [n, h, w] -> [n, -1] -> extend() n tensors
                    bs[i][m].extend(codeAtPartM.reshape(len(codeAtPartM), -1))
            pixels.append(h * w)
            torchvision.io.write_png(y[0].byte().cpu(), f"ckpt/images/test{i}_SSIM_{ssims[-1]}_PSNR_{psnrs[-1]}.png")

        encodeds, bpps = self._compress(bs, pixels)
        return {"ssim": sum(ssims) / len(ssims), "psnr": sum(psnrs) / len(psnrs), "bpp": sum(bpps) / len(bpps)}

    def _compress(self, codes: List[List[List[torch.Tensor]]], totalPixels: List[int]):
        encoder = ans.RansEncoder()
        compressed = list()
        bits = list()
        bpps = list()
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
            for i, codePerImage in enumerate(zip(*images)):
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
                bpps.append(8 * len(binary) / totalPixels[i])
        # # binary: 1 byte per word
        # # N * [binaries]
        # total = 8 * sum(len(binary) for binary in compressed)
        # bpp = float(total) / sum(totalPixels)

        # perfect = sum(bits) / sum(totalPixels)
        # self._logger.info("Estimate \"perfect\" BPP: %.4f", perfect)
        return compressed, bpps

    def _calculateFreq(self, code: torch.Tensor, k):
        count = torch.bincount(code.long(), minlength=k)
        return count / code.numel()
