import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from mcquic.utils.vision import RandomGamma, RandomPlanckianJitter, RandomAutocontrast, RandomHorizontalFlip, RandomVerticalFlip, PatchWiseErasing


# def hook(x):
#     print(type(x['jpg']))
#     exit()
#     return x['jpg']

def getTrainingPreprocess():
    return T.Compose([
        # T.ToTensor(),
        # T.Resize(512),
        T.RandomResizedCrop((512, 512), (0.75, 1.3333), (0.95, 1.05)),
        # T.ConvertImageDtype(torch.float32),
        RandomGamma()
    ])

def getTrainingPreprocessWithText():
    transform = T.Compose([
        T.ToTensor(),
        # T.Resize(512),
        T.RandomResizedCrop((512, 512), (0.75, 1.3333), (0.95, 1.05)),
        # T.ConvertImageDtype(torch.float32),
        RandomGamma()
    ])
    def call(x):
        img = x["jpeg"]
        text = x["label"]
        return {"jpeg": transform(img), "label": text}
    return call

def getTrainingTransform(gen: bool = False):
    return nn.Sequential(
        RandomPlanckianJitter(p=1.0),
        nn.Identity() if gen else RandomHorizontalFlip(p=0.5),
        nn.Identity() if gen else RandomVerticalFlip(p=0.5),
        T.Normalize(0.5, 0.5),
    )

def getTraingingPostprocess():
    return nn.Identity() # PatchWiseErasing()


def getEvalTransform():
    return T.Compose([
        T.ConvertImageDtype(torch.float32),
        AlignedCrop(256),
        T.Normalize(0.5, 0.5),
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
