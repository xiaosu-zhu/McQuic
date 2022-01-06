from typing import List

import torch
from torch import nn
from vlutils.metrics.meter import Handler

from mcqc.evaluation.metrics import Decibel, MsSSIM as M, PSNR as P


class MsSSIM(Handler):
    def __init__(self, format: str = r"%.2f dB"):
        super().__init__(format=format)
        self._msSSIM = nn.Sequential(
            Decibel(1.0),
            M(sizeAverage=False)
        )

    def to(self, device):
        self._msSSIM.to(device)
        return super().to(device)

    def handle(self, *, images: torch.ByteTensor, restored: torch.ByteTensor, **_) -> List[float]:
        # [N]
        results: torch.Tensor = self._msSSIM(images.float(), restored.float())
        return results.tolist()


class PSNR(Handler):
    def __init__(self, format: str = r"%.2f dB"):
        super().__init__(format=format)
        self._psnr = nn.Sequential(
            Decibel(255.0),
            P(sizeAverage=False)
        )

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

    def handle(self,*, images: torch.ByteTensor, binaries: List[bytes], **_) -> List[float]:
        bits = [len(b) * 8.0 for b in binaries]
        pixels = images.shape[-2] * images.shape[-1]
        bpps = [bit / pixels for bit in bits]
        return bpps
