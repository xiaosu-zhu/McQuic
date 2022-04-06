import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from mcquic.utils.vision import RandomGamma, RandomPlanckianJitter, RandomAutocontrast, RandomHorizontalFlip, RandomVerticalFlip, PatchWiseErasing

def getTrainingPreprocess():
    return T.Compose([
        T.RandomCrop(512, pad_if_needed=True),
        T.ConvertImageDtype(torch.float32),
        RandomGamma()
    ])

def getTrainingTransform():
    return nn.Sequential(
        RandomPlanckianJitter(p=1.0),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        T.Normalize(0.5, 0.5),
    )

def getTraingingPostprocess():
    return PatchWiseErasing()


def getEvalTransform():
    return T.Compose([
        T.ConvertImageDtype(torch.float32),
        AlignedCrop(128),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class AlignedCrop(nn.Module):
    def __init__(self, base: int = 128):
        super().__init__()
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        wCrop = w - w // self._base * self._base
        hCrop = h - h // self._base * self._base
        cropLeft = wCrop // 2
        cropRight = wCrop - cropLeft
        cropTop = hCrop // 2
        cropBottom = hCrop - cropTop

        if cropBottom == 0:
            cropBottom = -h
        if cropRight == 0:
            cropRight = -w

        x = x[..., cropTop:(-cropBottom), cropLeft:(-cropRight)]

        return x


class AlignedPadding(nn.Module):
    def __init__(self, base: int = 128):
        super().__init__()
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        wPadding = (w // self._base + 1) * self._base - w
        hPadding = (h // self._base + 1) * self._base - h

        wPadding = wPadding % self._base
        hPadding = hPadding % self._base

        padLeft = wPadding // 2
        padRight = wPadding - padLeft
        padTop = hPadding // 2
        padBottom = hPadding - padTop

        return F.pad(x, (padLeft, padRight, padTop, padBottom), "reflect")
