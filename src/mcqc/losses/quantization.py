from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

    def forward(self, images, restored, quantized, logits, latent):
        l2Loss = F.mse_loss(restored, images)
        l1Loss = F.l1_loss(restored, images)
        ssimLoss = 1 - self._msssim((restored + 1), (images + 1))
        regs = list()

        for logit in logits:
            # N, H, W, K -> N, HW, K
            posterior = Categorical(logits=logit)
            prior = Categorical(logits=torch.zeros_like(logit))
            reg = torch.distributions.kl_divergence(posterior, prior).mean()
            regs.append(reg)
        regs = sum(regs) / len(logits)
        return ssimLoss, l1Loss + l2Loss, regs
