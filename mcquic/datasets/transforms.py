import torch
from torch import nn
import math
from torchvision import transforms as T

from mcquic.utils.vision import RandomHorizontalFlip, RandomVerticalFlip, RandomGamma, RandomAutocontrast

def getTrainingTransform():
    return T.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAutocontrast(0.25),
        T.ConvertImageDtype(torch.float32),
        RandomGamma(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getTrainingPreprocess():
    return T.Compose([
        T.RandomCrop(512, pad_if_needed=True),
        T.RandomApply([T.RandomChoice([T.ColorJitter(0.4, 0, 0, 0), T.ColorJitter(0, 0.4, 0, 0), T.ColorJitter(0, 0, 0.4, 0), T.ColorJitter(0, 0, 0, 0.2), T.ColorJitter(0.4, 0.4, 0, 0), T.ColorJitter(0.4, 0, 0.4, 0), T.ColorJitter(0.4, 0, 0, 0.2), T.ColorJitter(0, 0.4, 0.4, 0), T.ColorJitter(0, 0.4, 0, 0.2), T.ColorJitter(0, 0, 0.4, 0.2), T.ColorJitter(0.4, 0.4, 0.4, 0), T.ColorJitter(0.4, 0.4, 0, 0.2), T.ColorJitter(0.4, 0, 0.4, 0.2), T.ColorJitter(0, 0.4, 0.4, 0.2), T.ColorJitter(0.4, 0.4, 0.4, 0.2)])], 0.25)
    ])

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
        _, h, w = x.shape
        wCrop = w - math.floor(w / self._base) * self._base
        hCrop = h - math.floor(h / self._base) * self._base
        cropLeft = wCrop // 2
        cropRight = wCrop - cropLeft
        cropTop = hCrop // 2
        cropBottom = hCrop - cropTop

        if cropBottom == 0:
            cropBottom = -h
        if cropRight == 0:
            cropRight = -w

        x = x[:, cropTop:(-cropBottom), cropLeft:(-cropRight)]

        return x
