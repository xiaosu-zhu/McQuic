from difflib import restore
import math
import torch
from torch import nn
import torch.nn.functional as F

from mcquic.validate.metrics import MsSSIM


class L1L2Loss(nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (F.mse_loss(input, target, reduction=self.reduction) + F.l1_loss(input, target, reduction=self.reduction)) / 2


class CompressionLossBig(nn.Module):
    def __init__(self, target):
        super().__init__()
        if target not in ["MsSSIM", "PSNR"]:
            raise ValueError(f"The argument `target` not in (\"MsSSIM\", \"PSNR\"), got \"{target}\".")
        if target == "MsSSIM":
            self._ssim = MsSSIM(data_range=2.0, sizeAverage=True)
            self._distortion = self._dSsim
        else:
            self._distortion = self._dPsnr

    def _dPsnr(self, restored, image):
        return (F.mse_loss(restored, image) + F.l1_loss(restored, image)) / 2

    def _dSsim(self, restored, image):
        return self._ssim(restored + 1, image + 1)

    def forward(self, restored, image, *_):
        dLoss = self._distortion(restored, image)
        return 0.0, dLoss
