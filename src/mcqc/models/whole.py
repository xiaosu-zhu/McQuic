import torch
from torch import nn

from mcqc.losses.structural import CompressionLoss

from .compressor import MultiScaleCompressor
from .discriminator import FullDiscriminator


class Whole(nn.Module):
    def __init__(self, k, channel):
        super().__init__()
        self._compressor = MultiScaleCompressor(k, channel)
        # self._discriminator = FullDiscriminator(channel // 4)

        self._cLoss = CompressionLoss()
        # self._ganLoss = nn.BCEWithLogitsLoss()

    def forward(self, step, image, temperature, hard):
        restored, codes, latents, logits, quantizeds = self._compressor(image, temperature, hard)
        # if step % 2 == 0:
        #     real = self._discriminator(image)
        #     fake = self._discriminator(restored.detach())
        #     rLabel = torch.empty_like(real).uniform_(0., 0.1)
        #     fLabel = torch.empty_like(fake).uniform_(0.9, 1.)
        #     lossR = self._ganLoss(real, rLabel)
        #     lossF = self._ganLoss(fake, fLabel)
        #     dLoss = lossF + lossR
        #     return (None, None, None, dLoss, None), (restored, codes, latents, logits, quantizeds)

        # fake = self._discriminator(restored)
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, codes, latents, logits, quantizeds)
        # gLoss = self._ganLoss(fake, torch.zeros_like(fake))
        return (ssimLoss, l1l2Loss, reg, None, None), (restored, codes, latents, logits, quantizeds)
