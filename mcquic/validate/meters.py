from typing import List, Tuple, Any
import itertools

import torch
from vlutils.metrics.meter import Handler

from mcquic.evaluation.metrics import Decibel, MsSSIM as M, PSNR as P


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
    def Result(self):
        # percentage of usage of all codes
        return self._temp

    def __str__(self) -> str:
        return "In Tensorboard."



class UniqueCodes(Handler):
    def __init__(self, k: List[int], *_):
        super().__init__("%2d/%2d")

        self._k = k
        self.accumulated: List[torch.Tensor] = list(torch.zeros([k]) for k in self._k)

    def reset(self):
        self.length = 0
        self.accumulated: List[torch.Tensor] = list(torch.zeros([k]) for k in self._k)

    def __call__(self, *args: Any, **kwds: Any):
        results = self.handle(*args, **kwds)

        # Only give stats of whole dataset
        self.length += 1

        for lv, unqiueCounts in enumerate(results):
            self.accumulated[lv] += unqiueCounts

    def handle(self, *, codes: List[torch.Tensor], **_) -> List[List[float]]:
        allCounts = list()
        # [n, m, h, w]
        for code, k in zip(codes, self._k):
            # [n, h, w]
            count = torch.bincount(code[:, 0].flatten(), minlength=k).cpu()
            allCounts.append(count)
        # lv * [each image's unique counts, only first group]
        return allCounts

    @property
    def Result(self) -> float:
        # percentage of usage of all codes
        return sum((float((acc > 0).sum()) / k) for acc, k in zip(self.accumulated, self._k)) / len(self._k)

    def __str__(self) -> str:
        return ", ".join(self._format % (sum(acc > 0), k) for (acc, k) in zip(self.accumulated, self._k))
