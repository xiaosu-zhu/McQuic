from logging import info
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import storch

from mcqc.losses.structural import CompressionLoss, QError
from mcqc.losses.mlm import MLELoss, MLMLoss, SAGLoss, ContextGANLoss, InfoMaxLoss
from mcqc.models.compressor import MultiScaleVQCompressor, PQCompressor, PQSAGCompressor, PQContextCompressor, PQCompressorFineTune
from mcqc.models.critic import SimpleCritic
from mcqc.models.discriminator import FullDiscriminator, LatentsDiscriminator
from mcqc.models.infoMax import InfoMax
from mcqc.utils.vision import Masking


class WholePQFineTune(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQCompressorFineTune(m, k, channel, numLayers)
        self._cLoss = CompressionLoss()
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, (quantized, latent), codes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, quantized, logits, latent)
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, None)


class WholePQ(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQCompressor(m, k, channel, numLayers)
        self._cLoss = CompressionLoss()
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, (quantized, latent), codes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, quantized, logits, latent)
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, None)


class WholePQContext(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQContextCompressor(m, k, channel, numLayers)
        self._cLoss = CompressionLoss()
        self._mLoss = ContextGANLoss()

    def forward(self, image, temp, step, **_):
        # maskedImage = self._masking(image)
        restored, codes, logits, predicts, targets = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        contextLoss = self._mLoss(predicts, targets, step)
        return (ssimLoss, l1l2Loss, contextLoss, reg), (restored, codes, predicts, logits, None)


class WholePQInfoMax(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQCompressor(m, k, channel, numLayers)
        self._discriminator = InfoMax(channel)
        self._cLoss = CompressionLoss()
        self._mLoss = InfoMaxLoss()

    def forward(self, image, temp, step, **_):
        # Y, T <- D(E(X))
        restored, (quantized, latent), codes, logits = self._compressor(image, temp, True)
        # minimize distortion
        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        # if ssimLoss > 0.1:
            # maximize I(Y; T)
            # logitsCondition, logitsJoint = self._discriminator(restored.detach(), quantized.detach())
        # else:
        logitsCondition, logitsJoint = self._discriminator(restored, quantized)
        infoLoss = self._mLoss(logitsCondition, logitsJoint, step)

        return (ssimLoss, l1l2Loss, infoLoss, reg), (restored, codes, None, logits, None)


class WholePQSAG(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQSAGCompressor(m, k, channel, numLayers)
        self._cLoss = CompressionLoss()
        self._mLoss = MLELoss()

    def forward(self, image, temp, **_):
        # maskedImage = self._masking(image)
        restored, (quantized, latent), codes, logits, predicts, targets = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, quantized, logits, None)
        mlmLoss = self._mLoss(predicts, targets)
        return (ssimLoss, l1l2Loss, mlmLoss, reg), (restored, codes, predicts, logits, latent)


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
