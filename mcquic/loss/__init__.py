from torch import nn
import torch.nn.functional as F

from mcquic.validate.metrics import MsSSIM as _m
from mcquic.validate.utils import Decibel
from mcquic.utils.registry import LossRegistry


class Distortion(nn.Module):
    def __init__(self, formatter: nn.Module):
        super().__init__()
        self._formatter = formatter

    def formatDistortion(self, loss):
        return self._formatter(loss)


class Rate(nn.Module):
    def __init__(self, formatter: nn.Module):
        super().__init__()
        self._formatter = formatter

    def formatRate(self, loss):
        return self._formatter(loss)


class BasicRate(Rate):
    def __init__(self, gamma: float = 1e-4):
        super().__init__(nn.Identity())
        self._gamma = gamma

    def _cosineLoss(self, codebook):
        losses = list()
        # m * [k, d]
        for c in codebook:
            # [k, k]
            pairwise = c @ c.T
            norm = (c ** 2).sum(-1)
            cos = pairwise / (norm[:, None] * norm).sqrt()
            cos.triu(1).sum()
            losses.append(cos)
        return sum(losses)

    def forward(self, logits, codebooks, *_):
        return sum(self._cosineLoss(codebook) for codebook in codebooks)


@LossRegistry.register
class MsSSIM(Distortion):
    def __init__(self):
        super().__init__(Decibel(1.0))
        self._ssim = _m(data_range=2.0, sizeAverage=True)

    def forward(self, restored, image, *_):
        return self._ssim(restored + 1, image + 1)

@LossRegistry.register
class PSNR(Distortion):
    def __init__(self):
        super().__init__(Decibel(2.0))

    def forward(self, restored, image, *_):
        return F.mse_loss(restored, image)
