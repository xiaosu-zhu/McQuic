import math
import torch
from torch import nn
import torch.nn.functional as F

from mcqc.evaluation.metrics import MsSSIM


class L1L2Loss(nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (F.mse_loss(input, target, reduction=self.reduction) + F.l1_loss(input, target, reduction=self.reduction)) / 2


class MeanAligning(nn.Module):
    def forward(self, quantized, code, codebook):
        count = code.sum((0, 1, 2))
        # [k, c], don't forget to divide N to get mean
        # [k, c] / [k, 1]
        meanQuantized = torch.einsum("nhwk,nhwc->kc", code, quantized) / count[:, None]
        meanQuantized = meanQuantized[count != 0]
        codebook = codebook[count != 0]
        print(meanQuantized.sum())
        alignLoss = F.mse_loss(codebook, meanQuantized)
        return alignLoss


class CodebookRegularization(nn.Module):
    def forward(self, codebook):
        # [k]
        inter = (codebook ** 2).sum(-1)
        # [k, k]
        intra = (codebook @ codebook.T).triu(1)

        loss = ((inter - 1.0) ** 2).mean() - intra.mean()

        return loss

class CodebookSpreading(nn.Module):
    def forward(self, codebook, temperature):
        k, d = codebook.shape
        # [k]
        inter = (codebook ** 2).sum(-1)
        # [k, k]
        intra = (codebook @ codebook.T).triu(1)

        indices = torch.ones_like(intra, dtype=torch.bool).triu(1)

        # [k, k] distance
        distance = (inter[:, None] - 2 * intra + inter)[indices]
        distance = distance[distance > 0]

        loss = F.relu(-((distance / d) * math.sqrt(k / temperature) + 1e-7).log()).mean()

        return loss

class L2Regularization(nn.Module):
    def forward(self, x, dim: int = -1):
        norm = (x ** 2).sum(dim)
        loss = ((norm - 1.0) ** 2).mean()
        return loss


class Regularization(nn.Module):
    def forward(self, logit):
        # [n, m, h, w, k]
        target = -math.log(logit.shape[-1])
        logit = logit.mean((2, 3))
        logit = logit - logit.logsumexp(-1, keepdim=True)
        # [n, m, k]
        prob = torch.softmax(logit, -1)
        t = prob * (logit - target)
        t[prob == 0] = 0
        return t.mean()


class CompressionLossBig(nn.Module):
    def __init__(self, target):
        super().__init__()
        if target not in ["ssim", "psnr"]:
            raise ValueError(f"The argument `target` not in (\"ssim\", \"psnr\"), got \"{target}\".")
        if target == "ssim":
            self._ssim = MsSSIM(data_range=2.0, sizeAverage=True)
            self._distortion = self._dSsim
        else:
            self._distortion = self._dPsnr

    def _dPsnr(self, restored, image):
        return F.mse_loss(restored, image)

    def _dSsim(self, restored, image):
        return self._ssim(restored + 1, image + 1)

    def forward(self, restored, image, *_):
        dLoss = self._distortion(restored, image)
        return dLoss
