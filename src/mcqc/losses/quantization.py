import math
from typing import List
import storch
from storch.wrappers import deterministic
import torch
from torch import nn
from torch._C import device
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn.functional as F
from torch.distributions import Categorical
from mcqc.consts import Consts

from mcqc.evaluation.metrics import MsSSIM


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake)))
    return d_loss


class QError(nn.Module):
    def forward(self, latents: List[torch.Tensor], zqs: List[torch.Tensor], softs: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for z, zq, soft in zip(latents, zqs, softs):
            qe = F.mse_loss(z.detach(), zq, reduction='none').mean(axis=(0, 2))
            commit = F.mse_loss(z, zq.detach(), reduction='none').mean(axis=(0, 2))
            softQE = F.mse_loss(z.detach(), soft, reduction='none').mean(axis=(0, 2))
            softCommit = F.mse_loss(z, soft.detach(), reduction='none').mean(axis=(0, 2))
            joint = F.mse_loss(soft, zq, reduction='none').mean(axis=(0, 2))
            loss += qe + softQE + 0.1 * joint # + 0.01 * commit + 0.1 * (softQE + 0.01 * softCommit)
        return loss


class CompressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._msssim = MsSSIM(data_range=2.0, sizeAverage=True)

    def forward(self, images, restored, codes, trueCodes, logits, z, zHat):
        device = images.device

        l2Loss = 0.0 # F.mse_loss(restored, images)
        l1Loss = 0.0 # F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim(restored + 1, images + 1)


        regs = list()
        n, h, w, k = logits[0].shape
        """
        # ths = torch.tensor(float(h * w) / k, device=device).clamp(1.0, h * w)

        # codes: [m, n, h, w]; logits: m * list(n, h, w, k); codeFreqMap: m * list([n, h, w]), binCounts: m * list([n, k])
        for code, logit, freqMap, binCount in zip(trueCodes.permute(1, 0, 2, 3), logits, codeFreqMap, binCounts):
            # binCount = binCount.float()
            # maxFreq, _ = binCount.max(-1, keepdim=True)

            # prob = ths / freqMap
            # Expectation of number of each bin not masked: (h * w) / k
            # maskProb = torch.distributions.Bernoulli(probs=(1.0 - prob).clamp(0.05, 0.95))
            maskProb = torch.distributions.Bernoulli(probs=(freqMap / float(h * w)).clamp(0.01, 0.99))
            # [n, h, w]
            needRegMask = maskProb.sample().bool()

            # # adjust final code
            # # fm, bc, regMask shape: [h, w]
            # for i, (c, fm, bc, regMask) in enumerate(zip(code, freqMap, binCount, needRegMask)):
            #     # needRegMask codes will be re-allocated
            #     newCount = torch.bincount(c[~regMask].flatten(), minlength=k)
            #     # contain negative numbers
            #     remain = ths - newCount
            #     # totally, have j negatives
            #     negativeTotalCount = (remain[remain < 0]).sum()
            #     # set negative to zero, then, we remain j numbers to remove
            #     remain[remain < 0] = 0
            #     # [h*w - sum(freqMap > ths)] sequence, the remaining codes to allocate
            #     sequence = torch.repeat_interleave(torch.arange(k, device=device), remain)
            #     # remove the last j numbers
            #     sequence = sequence[torch.randperm(len(sequence))]
            #     if negativeTotalCount < 0:
            #         sequence = sequence[:negativeTotalCount]
            #     code[i][regMask] = sequence
            # relaxedFreq = maxFreq # + binCount.mean(-1, keepdim=True)
            # reverse frequencies
            # max bin -> 0
            # min bin -> maxbin - minbin
            # [n, k]
            # reverseBin = relaxedFreq - binCount
            # [n, h, w], auto convert freq to prob in pytorch implementation
            # sample = torch.distributions.Categorical(probs=reverseBin).sample((h, w)).permute(2, 0, 1)
            # sample = torch.distributions.Categorical(logits=torch.zeros_like(logit)).sample()
            sample = torch.distributions.Categorical(probs=(binCount < 1).float()).sample((h, w)).permute(2, 0, 1)
            logit = logit.permute(0, 3, 1, 2)
            # [n, 1, 1], normalize then sigmoid, higher frequency -> higher weight
            # weight = freqMap / float(h * w) # ((freqMap / float(h * w) - 0.5)* 4).sigmoid()
            ceReg = F.cross_entropy(logit, sample, reduction="none") * needRegMask # * (float(h * w) / needRegMask.float().sum(dim=(1, 2), keepdims=True))
            # # cePush = 0.001 * F.cross_entropy(logit, code, reduction="none") * (~needRegMask)
            # cePush = F.cross_entropy(logit, code, reduction="none") * (~needRegMask) * weight
            regs.append((ceReg).mean())
        regs = sum(regs) / len(regs)
        """

        for logit, code in zip(logits, codes.permute(1, 0, 2, 3)):
            # [N, H, W, K] -> [N, K]
            # logit = logit.mean(dim=(1, 2))
            posterior = Categorical(logits=logit)
            # [N, H, W]
            reg = posterior.entropy().mean()
            # prior = Categorical(logits=torch.zeros_like(logit))
            # reg = torch.distributions.kl_divergence(posterior, prior).mean()
            # # [n, h, w, k]
            # # weight = (-logit).detach().softmax(-1)
            # # oneHot = F.one_hot(code, num_classes=logit.shape[-1]).float()
            # # [n, h, w]
            # # targetWeight = (weight * oneHot).sum(-1)
            # code = torch.randint(logit.shape[-1], [n, h, w], device=device)
            # logit = logit.permute(0, 3, 1, 2)
            # mle = F.cross_entropy(logit, code)
            # # regs.append(mle)
            # regs.append(reg + 0.01 * mle)
        regs = sum(regs) / len(logits)
        # regs = 0.0
        return ssimLoss, l1Loss + l2Loss, regs


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


class CodebookSpreading(nn.Module):
    def forward(self, codebook):
        # [k]
        inter = (codebook ** 2).sum(-1)
        # [k, k]
        intra = (codebook @ codebook.T).triu(1)

        loss = ((inter - 1.0) ** 2).mean() - intra.mean()

        return loss

class L2Regularization(nn.Module):
    def forward(self, x, dim: int = -1):
        norm = (x ** 2).sum(dim)
        loss = ((norm - 1.0) ** 2).mean()
        return loss


class Regularization(nn.Module):
    def forward(self, logit):
        target = -math.log(logit.shape[-1])
        logit = logit.sum((0, 1, 2))
        logit = logit - logit.logsumexp(-1)
        # [k]
        prob = torch.softmax(logit, -1)
        t = prob * (logit - target)
        t[prob == 0] = 0
        return t.sum()



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

    def _dPsnr(self, image, restored):
        return F.mse_loss(restored, image)

    def _dSsim(self, image, restored):
        return 1 - self._ssim(restored + 1, image + 1)

    def forward(self, images, restored):
        dLoss = self._distortion(restored, images)
        return dLoss


class CompressionLossQ(nn.Module):
    def __init__(self):
        super().__init__()
        self._msssim = MsSSIM(data_range=2.0, sizeAverage=True)

    def forward(self, images, restored, trueCodes, logits):
        device = images.device

        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim(restored + 1, images + 1)

        # [n, k, h, w, m]
        logits = logits.permute(0, 4, 1, 2, 3)
        fakeCodes = torch.randint_like(trueCodes, logits.shape[1])

        regs = F.cross_entropy(logits, fakeCodes)

        # ssimLoss = (1 - self._msssim((restored + 1), (images + 1))).log10().mean()
        # ssimLoss = -F.binary_cross_entropy(ssimLoss, torch.ones_like(ssimLoss))

        # regs = list()

        # for logit, code in zip(logits, trueCodes.permute(1, 0, 2, 3)):
        #     # [N, H, W, K] -> [N, K]
        #     # logit = logit.mean(dim=(1, 2))
        #     posterior = Categorical(logits=logit)
        #     prior = Categorical(logits=torch.zeros_like(logit))
        #     reg = torch.distributions.kl_divergence(posterior, prior).mean()
        #     # [n, h, w, k]
        #     # weight = (-logit).detach().softmax(-1)
        #     # oneHot = F.one_hot(code, num_classes=logit.shape[-1]).float()
        #     # [n, h, w]
        #     # targetWeight = (weight * oneHot).sum(-1)
        #     code = torch.randint_like(code, logit.shape[-1])
        #     logit = logit.permute(0, 3, 1, 2)
        #     mle = F.cross_entropy(logit, code)
        #     regs.append(reg + 0.01 * mle)
        # regs = sum(regs) / len(logits)
        # regs = 0.0
        return ssimLoss, l1Loss + l2Loss, regs


class CompressionLossNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._msssim = MsSSIM(data_range=2.0, sizeAverage=True)

    def forward(self, images, restored, logit):
        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim(restored + 1, images + 1)
        # ssimLoss = (1 - self._msssim((restored + 1), (images + 1))).log10().mean()
        # ssimLoss = -F.binary_cross_entropy(ssimLoss, torch.ones_like(ssimLoss))

        # N, M, K, H, W -> N, M, H, W, K
        logit = logit.permute(0, 1, 3, 4, 2)
        posterior = Categorical(logits=logit)
        prior = Categorical(logits=torch.zeros_like(logit))
        reg = torch.distributions.kl_divergence(posterior, prior).mean()
        return ssimLoss, l1Loss + l2Loss, reg


class CompressionLossS(nn.Module):
    def __init__(self):
        super().__init__()
        self._msssim = MsSSIM(data_range=2.0, sizeAverage=False)

    @deterministic(flatten_plates=True)
    def ff(self, images, restored, logits):
        images = images[..., 0]
        restored = restored[..., 0]
        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim((restored + 1), (images + 1))
        regs = list()

        # N, H, W, K -> N, HW, K
        posterior = Categorical(logits=logits)
        prior = Categorical(logits=torch.zeros_like(logits))
        reg = torch.distributions.kl_divergence(posterior, prior)
        reg = reg.sum(dim=(1,2,3))
        return ssimLoss, reg

    def forward(self, images, restored, quantized, logits, latent):
        return self.ff(images, restored, logits)
