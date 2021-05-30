import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import storch

from mcqc.losses.structural import CompressionLoss, QError
from mcqc.losses.mlm import MLMLoss, SAGLoss, ContextGANLoss
from mcqc.models.compressor import MultiScaleCompressor, MultiScaleVQCompressor, MultiScaleCompressorRein, MultiScaleCompressorStorch, MultiScaleCompressorExp, MultiScaleCompressorSplitted, PQCompressor, PQSAGCompressor, PQContextCompressor
from mcqc.models.critic import SimpleCritic
from mcqc.models.discriminator import FullDiscriminator, LatentsDiscriminator
from mcqc.utils.vision import Masking


class WholePQ(nn.Module):
    def __init__(self, k, channel, numLayers):
        super().__init__()
        self._compressor = PQCompressor(k, channel, numLayers)
        self._cLoss = CompressionLoss()
        self._qLoss = QError()

    def forward(self, image, temp, **_):
        restored, codes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, None, logits, None)


class WholePQContext(nn.Module):
    def __init__(self, k, channel, numLayers):
        super().__init__()
        self._compressor = PQContextCompressor(k, channel, numLayers)
        self._cLoss = CompressionLoss()
        self._mLoss = ContextGANLoss()

    def forward(self, image, temp, step, **_):
        # maskedImage = self._masking(image)
        restored, codes, logits, predicts, targets = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        contextLoss = self._mLoss(predicts, targets, step)
        return (ssimLoss, l1l2Loss, contextLoss, reg), (restored, codes, predicts, logits, None)


class WholePQSAG(nn.Module):
    def __init__(self, k, channel, numLayers):
        super().__init__()
        self._compressor = PQSAGCompressor(k, channel, numLayers)
        self._cLoss = CompressionLoss()
        self._mLoss = SAGLoss()

    def forward(self, image, temp, **_):
        # maskedImage = self._masking(image)
        restored, codes, logits, predicts, targets = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        mlmLoss = self._mLoss(predicts, targets)
        return (ssimLoss, l1l2Loss, mlmLoss + 1e-5 * reg), (restored, codes, predicts, logits, None)


class Whole(nn.Module):
    def __init__(self, k, channel, numLayers):
        super().__init__()
        self._compressor = MultiScaleCompressor(k, channel, numLayers)
        # self._discriminator = FullDiscriminator(channel // 4)

        self._cLoss = CompressionLoss()
        self._qLoss = QError()

    def forward(self, image, maskProb, temp):
        restored, codes, latents, logits, quantizeds = self._compressor(image, maskProb, temp, True)
        # if step % 2 == 0:
        #     real = self._discriminator(image.detach())
        #     fake = self._discriminator(restored.detach())
        #     dLoss = hinge_d_loss(real, fake)
        #     return (None, None, None, dLoss, None), (restored, codes, latents, logits, quantizeds)

        # fake = self._discriminator(restored)
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, latents, logits, quantizeds)
        # qError = self._qLoss(latents, codebooks, logits, codes)
        # gLoss = -1 * fake.mean()
        return (ssimLoss, l1l2Loss, reg), (restored, codes, latents, logits, quantizeds)


class WholeTwoStage(nn.Module):
    def __init__(self, k, channel, nPreLayers):
        super().__init__()
        self._compressor = MultiScaleCompressorSplitted(k, channel, nPreLayers)
        # self._discriminator = FullDiscriminator(channel // 4)
        self._cLoss = CompressionLossTwoStage()
        self._qLoss = QError()

    # @property
    # def codebook(self):
    #     return self._compressor._quantizer._codebook0

    # @torch.cuda.amp.autocast()
    def forward(self, image, temp, e2e):
        restored, codes, latents, logits, quantizeds, softQs = self._compressor(image, temp, e2e)
        ssimLoss, l1l2Loss, qLoss, reg = self._cLoss(image, restored, latents, logits, quantizeds, softQs)
        return (ssimLoss, l1l2Loss, qLoss, reg), (restored, codes, latents, logits, quantizeds)


class WholeTwoStageWithGan(nn.Module):
    def __init__(self, k, channel, nPreLayers):
        super().__init__()
        self._compressor = MultiScaleCompressor(k, channel, nPreLayers)
        self._discriminator = LatentsDiscriminator(channel)
        self._cLoss = CompressionLossTwoStageWithGan()
        self._qLoss = QError()

    # @property
    # def codebook(self):
    #     return self._compressor._quantizer._codebook0

    # @torch.cuda.amp.autocast()
    def forward(self, image, temp, e2e, step):
        restored, codes, latents, logits, quantizeds = self._compressor(image, temp, e2e)
        if step % 2 == 0:
            real = self._discriminator(latents[0].detach())
            fake = self._discriminator(quantizeds[0].detach())
            ssimLoss, l1l2Loss, qLoss, gLoss, dLoss, reg = self._cLoss(image, restored, latents, logits, quantizeds, real, fake, step)
            return (ssimLoss, l1l2Loss, qLoss, gLoss, dLoss, reg), (restored, codes, latents, logits, quantizeds)
        real = None
        fake = self._discriminator(quantizeds[0])
        ssimLoss, l1l2Loss, qLoss, gLoss, dLoss, reg = self._cLoss(image, restored, latents, logits, quantizeds, real, fake, step)
        return (ssimLoss, l1l2Loss, qLoss, gLoss, dLoss, reg), (restored, codes, latents, logits, quantizeds)


class WholeStorch(nn.Module):
    def __init__(self, k, channel, nPreLayers):
        super().__init__()
        self._compressor = MultiScaleCompressorStorch(k, channel, nPreLayers)
        # self._discriminator = FullDiscriminator(channel // 4)

        self._cLoss = CompressionLoss()
        self._qLoss = QError()

    # @property
    # def codebook(self):
    #     return self._compressor._quantizer._codebook0

    def forward(self, image, temp, transform, cv):
        image = storch.denote_independent(image, 0, "data")
        restored, codes, latents, logits, quantizeds = self._compressor(image, temp, transform)
        # if step % 2 == 0:
        #     real = self._discriminator(image.detach())
        #     fake = self._discriminator(restored.detach())
        #     dLoss = hinge_d_loss(real, fake)
        #     return (None, None, None, dLoss, None), (restored, codes, latents, logits, quantizeds)

        # fake = self._discriminator(restored)
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, latents, logits, quantizeds, cv)
        storch.add_cost(ssimLoss + l1l2Loss + reg, "reconstruction")
        # qError = self._qLoss(latents, codebooks, logits, codes)
        # gLoss = -1 * fake.mean()
        return (ssimLoss, l1l2Loss, reg), (restored, codes, latents, logits, quantizeds)


class WholeRein(nn.Module):
    def __init__(self, k, channel, nPreLayers):
        super().__init__()
        self._compressor = MultiScaleCompressorRein(k, channel, nPreLayers)
        self._critic = SimpleCritic(channel)
        self._reward = CompressionReward()
        self._k = k

    # @property
    # def codebook(self):
    #     return self._compressor._quantizer._codebook0

    def forward(self, image, codes=None):
        if codes is not None:
            quantizeds, logits, negLogPs = self._compressor(image, codes)
            values = self._critic(quantizeds)
            return logits, negLogPs, values

        restored, codes, latents, negLogPs, logits, quantizeds = self._compressor(image)
        values = self._critic(quantizeds)
        # for code in codes:
        #     _, count = torch.unique(code, False, return_counts=True, dim=0)

        ssim, psnr, ssimLoss, l1l2Loss = self._reward(image, restored)

        reward = ssim + (psnr / 5)

        # qError = self._qLoss(latents, codebooks, logits, codes)
        # gLoss = -1 * fake.mean()
        return ssimLoss, l1l2Loss, [reward], restored, codes, latents, negLogPs, logits, quantizeds, values


class WholeVQ(nn.Module):
    def __init__(self, k, channel, nPreLayers):
        super().__init__()
        self._compressor = MultiScaleVQCompressor(k, channel, nPreLayers)
        # self._discriminator = FullDiscriminator(channel // 4)

        self._cLoss = CompressionLoss()
        self._qLoss = QError()

    @property
    def codebook(self):
        return self._compressor._quantizer._codebook

    def forward(self, step, image, temperature, hard, cv):
        restored, codes, latents, (zs, zq, softs), quantizeds, codebooks = self._compressor(image, temperature, hard)
        # if step % 2 == 0:
        #     real = self._discriminator(image.detach())
        #     fake = self._discriminator(restored.detach())
        #     dLoss = hinge_d_loss(real, fake)
        #     return (None, None, None, dLoss, None), (restored, codes, latents, logits, quantizeds)

        # fake = self._discriminator(restored)
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, codebooks, latents, None, quantizeds, cv)
        qError = self._qLoss(zs, zq, softs)
        # gLoss = -1 * fake.mean()
        return (ssimLoss, l1l2Loss, qError), (restored, codes, latents, None, quantizeds)
