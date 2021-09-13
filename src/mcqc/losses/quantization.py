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


class CompressionLossBig(nn.Module):
    def __init__(self):
        super().__init__()
        self._msssim = MsSSIM(data_range=2.0, sizeAverage=True)

    def forward(self, images, restored, c1, c2, predict, l1, l2):
        device = images.device

        _, _, h, w = c1.shape

        l2Loss = 0.0 # F.mse_loss(restored, images)
        l1Loss = 0.0 # F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim(restored + 1, images + 1)
        # [N, K, M, H, W], [N, M, H, W]
        contextLoss = F.cross_entropy(predict, c1)
        # [n, m, h, w, k] -> [n, k, m, h, w]
        # l1, l2 = l1.permute(0, 4, 1, 2, 3), l2.permute(0, 4, 1, 2, 3)
        # [N, K, M, H, W], [N, M, H, W]
        # sum(-logP) / ()
        # bppLoss = (F.cross_entropy(l1, c1, reduction="mean") + F.cross_entropy(l2, c2, reduction="mean")) / math.log(2)

        l1 = l1.mean((2,3))
        l2 = l2.mean((2,3))

        posterior1 = torch.distributions.Categorical(logits=l1)
        posterior2 = torch.distributions.Categorical(logits=l2)

        prior1 = torch.distributions.Categorical(logits=torch.zeros_like(l1))
        prior2 = torch.distributions.Categorical(logits=torch.zeros_like(l2))

        reg = torch.distributions.kl_divergence(posterior1, prior1).mean() + torch.distributions.kl_divergence(posterior2, prior2).mean()

        return ssimLoss, contextLoss, reg


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
