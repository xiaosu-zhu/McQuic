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


@LossRegistry.register
class MsSSIM(Distortion):
    def __init__(self):
        super().__init__(Decibel(1.0))
        self._ssim = _m(data_range=2.0, sizeAverage=True)

    def forward(self, restored, image, *_):
        return 0.0, self._ssim(restored + 1, image + 1)

@LossRegistry.register
class PSNR(Distortion):
    def __init__(self):
        super().__init__(Decibel(2.0))

    def forward(self, restored, image, *_):
        return 0.0, F.mse_loss(restored, image)
