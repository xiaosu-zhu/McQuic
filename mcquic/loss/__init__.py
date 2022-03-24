from torch import nn
import torch.nn.functional as F

from mcquic.validate.metrics import MsSSIM as _m
from mcquic.validate.utils import Decibel
from mcquic.utils.registry import LossRegistry


class Distortion(nn.Module):
    def __init__(self, target):
        super().__init__()
        if target not in ["MsSSIM", "PSNR"]:
            raise ValueError(f"The argument `target` not in (\"MsSSIM\", \"PSNR\"), got \"{target}\".")
        if target == "MsSSIM":
            self._ssim = _m(data_range=2.0, sizeAverage=True)
            self._distortion = self._dSsim
        else:
            self._distortion = self._dPsnr

        self._formatter = Decibel(1.0 if target == "MsSSIM" else 2.0)

    def _dPsnr(self, restored, image):
        return F.mse_loss(restored, image)

    def _dSsim(self, restored, image):
        return self._ssim(restored + 1, image + 1)

    def forward(self, restored, image, *_):
        dLoss = self._distortion(restored, image)
        return 0.0, dLoss

    def formatDistortion(self, loss):
        return self._formatter(loss)


@LossRegistry.register
class MsSSIM(Distortion):
    pass


@LossRegistry.register
class PSNR(Distortion):
    pass
