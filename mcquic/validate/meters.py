from typing import List
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

    def handle(self, *, images: torch.ByteTensor, binaries: List[List[bytes]], **_) -> List[float]:
        bits = [len(bi) * 8 for bi in itertools.chain(*binaries)]
        pixels = images.shape[-2] * images.shape[-1]
        bpps = [bit / pixels for bit in bits]
        return bpps
