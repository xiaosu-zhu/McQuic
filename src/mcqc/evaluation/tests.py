import abc
from typing import Dict, List

from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mcqc.config import Config
from mcqc.evaluation.helpers import psnr
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans

from mcqc.losses.ssim import MsSSIM, ssim


def deTrans(image):
    return ((image * 0.5 + 0.5) * 255).clamp(0.0, 255.0).byte()


class Test(abc.ABC):
    def __init__(self, config: Config, encoder: nn.Module, decoder: nn.Module, device: int, **kwArgs):
        super().__init__()
        self._device = device
        self._config = config
        self._encoder = encoder.to(device)
        self._decoder = decoder.to(device)
        self._encoder.eval()
        self._decoder.eval()
        torch.autograd.set_grad_enabled(False)
        setattr(self, "test", torch.inference_mode()(self.test))

    @abc.abstractmethod
    def test(self) -> Dict[str, float]:
        raise NotImplementedError


class Speed(Test):
    def __init__(self, **kwArgs) -> None:
        super().__init__(**kwArgs)
        # same as kodak
        self._testInput = torch.rand(24, 3, 768, 512).to(self._device)

    def test(self):
        # warmup
        for _ in trange(10):
            self._decoder(*self._encoder(self._testInput))
        torch.cuda.synchronize()

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test encoder
        for _ in trange(100):
            encoderOutputs = self._encoder(self._testInput)
        endEvent.record()
        torch.cuda.synchronize()
        encoderMs = startEvent.elapsed_time(endEvent) / (100 * len(self._testInput))

        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)

        startEvent.record()
        # test encoder
        for _ in trange(100):
            self._decoder(*encoderOutputs)
        endEvent.record()
        torch.cuda.synchronize()
        decoderMs = startEvent.elapsed_time(endEvent) / (100 * len(self._testInput))

        return {"encoderForwardTime": encoderMs, "decoderForwardTime": decoderMs}


class Performance(Test):
    def __init__(self, dataset: Dataset, **kwArgs):
        super().__init__(**kwArgs)
        self._dataLoader = DataLoader(dataset, pin_memory=True)
        self._ssim = MsSSIM(size_average=False).to(self._device)

    def test(self):
        ssims = list()
        psnrs = list()
        bs = list()
        pixels = list()
        for x in self._dataLoader:
            x = x.to(self._device, non_blocking=True)
            _, _, h, w = x.shape
            # [1, ?, ?, m]
            b = self._encoder(x)
            y = self._decoder(b)
            x, y = deTrans(x), deTrans(y)
            ssims.append(float(-(1.0 - self._ssim(x, y)).log10()))
            psnrs.append(float(psnr(x, y)))
            # list of [m, ?]
            bs.append(b.permute(0, 3, 1, 2).reshape(b.shape[-1], -1))
            pixels.append(h * w)

        encodeds, bpps = self._compress(bs, pixels)
        return {"ssim": sum(ssims) / len(ssims), "psnr": sum(psnrs) / len(psnrs), "bpp": sum(bpps) / len(bpps)}

    def _compress(self, codes: List[torch.Tensor], pixels: List[int]):
        compressed = list()
        bpps = list()
        cdfs = list()
        m = codes[0].shape[0]
        for i in range(m):
            # b: cat list of [M, ?] -> [?]
            b = torch.cat([x[i] for x in codes])
            # list of 256 probs
            prob = self._calculateFreq(b, self._config.Model.k)
            cdf = pmf_to_quantized_cdf(prob.tolist(), 16)
            # M * [cdf]
            cdfs.append(cdf)
        encoder = ans.RansEncoder()
        # codePerImage: [m, ?]
        for pix, codePerImage in zip(pixels, codes):
            # params: List of symbols, List of indices of pdfs, List of pdfs, List of upper-bounds, List of offsets
            # [0, 1, 2, 3], [0, 0, 1, 1], [[xx, xx, xx, xx], [xx, xx, xx, xx]], [4, 4, 4, 4], [0, 0, 0, 0]
            binary = encoder.encode_with_indexes(codePerImage.flatten().int().tolist(), torch.arange(codePerImage.shape[0])[:, None, None].expand_as(codePerImage).flatten().int().tolist(), cdfs, [self._config.Model.k] * self._config.Model.m, torch.zeros_like(codePerImage).flatten().int().tolist())
            compressed.append(binary)
            # binary: 1 byte per word
            bpps.append(float(8 * len(binary)) / pix)
        return compressed, bpps

    def _calculateFreq(self, code: torch.Tensor, k):
        count = torch.bincount(code, minlength=k)
        return count / code.numel()
