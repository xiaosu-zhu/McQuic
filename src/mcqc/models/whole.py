import torch
from torch import nn
import torch.nn.functional as F

from mcqc.losses.structural import CompressionLoss

from .compressor import MultiScaleCompressor
from .discriminator import FullDiscriminator


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class Whole(nn.Module):
    def __init__(self, k, channel):
        super().__init__()
        self._compressor = MultiScaleCompressor(k, channel)
        # self._discriminator = FullDiscriminator(channel // 4)

        self._cLoss = CompressionLoss()

    def forward(self, step, image, temperature, hard):
        restored, codes, latents, logits, quantizeds = self._compressor(image, temperature, hard)
        # if step % 2 == 0:
        #     real = self._discriminator(image.detach())
        #     fake = self._discriminator(restored.detach())
        #     dLoss = hinge_d_loss(real, fake)
        #     return (None, None, None, dLoss, None), (restored, codes, latents, logits, quantizeds)

        # fake = self._discriminator(restored)
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, codes, latents, logits, quantizeds)
        # gLoss = -1 * fake.mean()
        return (ssimLoss, l1l2Loss, reg, None, None), (restored, codes, latents, logits, quantizeds)
