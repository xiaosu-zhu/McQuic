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

from mcqc.evaluation.metrics import MsSSIM, ssim


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

    def forward(self, images, restored, codes, logits, codeFreqMap, binCounts):
        device = images.device

        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim(restored + 1, images + 1)
        # ssimLoss = (1 - self._msssim((restored + 1), (images + 1))).log10().mean()
        # ssimLoss = -F.binary_cross_entropy(ssimLoss, torch.ones_like(ssimLoss))
        regs = list()

        n, h, w, k = logits[0].shape
        ths = torch.tensor((h * w) // k, dtype=torch.long, device=device)

        # codes: [m, n, h, w]; logits: m * list(n, h, w, k); codeFreqMap: m * list([n, h, w]), binCounts: m * list([n, k])
        for code, logit, freqMap, binCount in zip(codes.permute(1, 0, 2, 3), logits, codeFreqMap, binCounts):


            # Expectation of number of each bin not masked: (h * w) / k
            maskProb = torch.distributions.Bernoulli(probs=(1.0 - float(h * w) / (freqMap * k)).clamp(0.0, 1.0))
            # maxFreq, _ = binCount.max(-1, keepdim=True)
            # [n, h, w]
            needRegMask = maskProb.sample().bool()

            code = code.clone().detach()

            # adjust final code
            # fm, bc, regMask shape: [h, w]
            for i, (c, fm, bc, regMask) in enumerate(zip(code, freqMap, binCount, needRegMask)):
                # needRegMask codes will be re-allocated
                newCount = torch.bincount(c[~regMask].flatten(), minlength=k)
                # contain negative numbers
                remain = ths - newCount
                # totally, have j negatives
                negativeTotalCount = (remain[remain < 0]).sum()
                # set negative to zero, then, we remain j numbers to remove
                remain[remain < 0] = 0
                # [h*w - sum(freqMap > ths)] sequence, the remaining codes to allocate
                sequence = torch.repeat_interleave(torch.arange(k, device=device), remain)
                # remove the last j numbers
                sequence = sequence[torch.randperm(len(sequence))]
                if negativeTotalCount < 0:
                    sequence = sequence[:negativeTotalCount]
                code[i][regMask] = sequence

            bc = torch.bincount(code.flatten())

            # relaxedFreq = maxFreq + binCount.mean(-1, keepdim=True)
            # # reverse frequencies
            # # max bin -> meanFreq
            # # min bin -> meanFreq + maxbin - minbin
            # # [n, k]
            # reverseBin = relaxedFreq - binCount
            # # [n, h, w], auto convert freq to prob in pytorch implementation
            # sample = torch.distributions.Categorical(logits=torch.zeros_like(logit)).sample()
            logit = logit.permute(0, 3, 1, 2)
            # # [n, 1, 1], normalize then sigmoid, higher frequency -> higher weight
            # weight = freqMap / float(h * w) # ((freqMap / float(h * w) - 0.5)* 4).sigmoid()
            ceReg = F.cross_entropy(logit, code, reduction="none") * needRegMask
            cePush = 0.0001 * F.cross_entropy(logit, code, reduction="none") * (~needRegMask)
            # cePush = F.cross_entropy(logit, code, reduction="none") * (1 - needRegMask) * weight
            regs.append((ceReg + cePush).mean())
        # # [m, n, h, w] and m * list(n, h, w, k) logits and [n, k] frequencies
        # for code, logit, freq in zip(codes.permute(1, 0, 2, 3), logits, codeFreq):
        #     # perturb code by the most rare codes with 0.1 probability
        #     p = 1. / freq.shape[-1]
        #     freq[freq < 1.] = 1.
        #     weight = p / (freq + 1e-6)
        #     dropoutMask = torch.distributions.Bernoulli(probs=torch.tensor(0.1, device=logit.device)).sample((n, h, w))
        #     # [n, k] probs sample (h, w) -> [n, h, w]
        #     sample = torch.distributions.Categorical(probs=weight).sample(((dropoutMask > 0.5).int().sum(), ))
        #     code[dropoutMask > 0.5] = sample
        #     # input: [n, k, h, w], target: [n, h, w] -> [n, h, w]
        #     ce = F.cross_entropy(logit.permute(0, 3, 1, 2), code, reduction="none")
        #     # [n, k]
        #     weight = torch.zeros((n, k), device=ce.device)
        #     for i, c in enumerate(code):
        #         # frequency for all k entries --- sum up to h * w
        #         frequency = torch.bincount(c.flatten(), minlength=k)
        #         p = 1. / len(torch.unique(c.flatten()))
        #         # weights, higher frequency, lower weights
        #         weight[i] = p / (frequency + 1e-6)
        #     # indexing frequency by code to get per code weight
        #     ix = torch.arange(n, device=logit.device)[:, None, None].expand_as(code)
        #     # [n, h, w] weight sum to 1 every row, peer-to-peer weighting `ce` to balance the loss
        #     weight = weight[[ix, code]]
        #     ce = (ce * weight).mean()
        #     regs.append(ce)

        regs = sum(regs) / len(regs)

        # for logit in logits:
        #     # [N, H, W, K] -> [N, K]
        #     # logit = logit.mean(dim=(1, 2))
        #     posterior = Categorical(logits=logit)
        #     prior = Categorical(logits=torch.zeros_like(logit))
        #     reg = torch.distributions.kl_divergence(posterior, prior).mean()
        #     regs.append(reg)
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
