from typing import List, Tuple, Any, Union

import torch
from torch import nn
from vlutils.metrics.meter import Handler

from .metrics import MsSSIM as M, PSNR as P
from .utils import Decibel


class MsSSIM(Handler):
    def __init__(self, format: str = r"%.2f dB"):
        super().__init__(format=format)
        self._msSSIM = M(sizeAverage=False)
        self._formatter = Decibel(1.0)

    def to(self, device):
        self._msSSIM.to(device)
        return super().to(device)

    def handle(self, *, images: torch.ByteTensor, restored: torch.ByteTensor, **_) -> List[float]:
        # [N]
        results: torch.Tensor = self._formatter(self._msSSIM(images.float(), restored.float()))
        return results.tolist()


class PSNR(Handler):
    def __init__(self, format: str = r"%.2f dB"):
        super().__init__(format=format)
        self._psnr = P(sizeAverage=False)

    def to(self, device):
        self._psnr.to(device)
        return super().to(device)

    def handle(self, *, images: torch.ByteTensor, restored: torch.ByteTensor, **_) -> List[float]:
        # [N]
        results: torch.Tensor = self._psnr(images.float(), restored.float())
        return results.tolist()


class BPP(Handler):
    def __init__(self, format: str = r"%.4f"):
        super().__init__(format=format)

    @staticmethod
    def bitLength(byteArray: List[bytes]):
        return sum(len(bi) * 8 for bi in byteArray)

    def handle(self, *, images: torch.ByteTensor, binaries: List[List[bytes]], **_) -> List[float]:
        # binaries: List of binary, len = n, len(binaries[0]) = level
        bits = [self.bitLength(bis) for bis in binaries]
        pixels = images.shape[-2] * images.shape[-1]
        bpps = [bit / pixels for bit in bits]
        return bpps


class Visualization(Handler):
    def __init__(self):
        super().__init__()
        self._temp = None

    def reset(self):
        self._temp = None

    def __call__(self, *args: Any, **kwds: Any):
        self._temp = self.handle(*args, **kwds)

    def handle(self, *, restored: torch.ByteTensor, **_) -> torch.Tensor:
        # binaries: List of binary, len = n, len(binaries[0]) = level
        return restored.detach()

    @property
    def ShowInSummary(self) -> bool:
        return False

    @property
    def Result(self):
        # percentage of usage of all codes
        return self._temp

    def __str__(self) -> str:
        return "In Tensorboard."


class IdealBPP(Handler):
    def __init__(self, k: List[int], format: str = r"%.4f"):
        super().__init__(format)

        self._k = k
        self.accumulated: List[torch.Tensor] = list(torch.zeros((k)) for k in self._k)
        self.totalPixels = torch.zeros(())
        self.totalCodes: List[torch.Tensor] = list(torch.zeros(()) for k in self._k)

    def reset(self):
        self.length = 0
        self.accumulated = list(torch.zeros((k)) for k in self._k)
        self.totalPixels = torch.zeros(())
        self.totalCodes = list(torch.zeros(()) for k in self._k)

    def __call__(self, *args: Any, **kwds: Any):
        results, pixels, codes = self.handle(*args, **kwds)

        # Only give stats of whole dataset
        self.length += 1

        for lv, unqiueCounts in enumerate(results):
            self.accumulated[lv] += unqiueCounts
        self.totalPixels += pixels
        for lv, codeCount in enumerate(codes):
            self.totalCodes[lv] += codeCount

    def handle(self, *, codes: List[torch.Tensor], images: torch.ByteTensor, **_) -> Tuple[List[torch.Tensor], int, List[int]]:
        allCounts: List[torch.Tensor] = list()
        codesNum: List[int] = list()
        # [n, m, h, w]
        for code, k in zip(codes, self._k):
            # [n, h, w]
            count = torch.bincount(code[:, 0].flatten(), minlength=k).cpu()
            allCounts.append(count)
            codesNum.append(code.numel())
        # lv * [each image's unique counts, only first group]
        return allCounts, images.numel(), codesNum

    @property
    def Result(self) -> float:
        totalBits = 0
        for codeUsage, codeCount in zip(self.accumulated, self.totalCodes):
            prob = codeUsage / codeUsage.sum()
            estimateEntropy = prob.log2()
            estimateEntropy[estimateEntropy == float("-inf")] = 0
            estimateEntropy = -(prob * estimateEntropy).sum()
            totalBits += float(estimateEntropy) * float(codeCount)
        # percentage of usage of all codes
        return totalBits / float(self.totalPixels)

    def __str__(self) -> str:
        return self._format % self.Result
