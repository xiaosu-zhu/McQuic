from numpy import imag
from torch import nn
import torch
import storch

from mcqc.losses.quantization import CompressionLoss, CompressionLossNew, QError
from mcqc.models.compressor import AQCompressor, PQCompressor, PQCompressorNew, PQContextCompressor, PQRelaxCompressor, PQCompressorTwoPass


class WholePQ(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._compressor = PQCompressor(m, k, channel, withGroup, withAtt, withDropout, alias)
        self._cLoss = CompressionLoss()
        self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, (quantized, latent), codes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, quantized, logits, latent)
        self._movingMean -= 0.9 * (self._movingMean - ssimLoss.mean())
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, None)


class WholeAQ(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._compressor = AQCompressor(m, k, channel, withGroup, withAtt, withDropout, alias, ema)
        self._cLoss = CompressionLoss()
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, (quantized, latent), (codes, frequencyMaps, binCounts, trueCodes), logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, trueCodes, logits, frequencyMaps, binCounts)
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, None)


class WholePQNew(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._compressor = PQCompressorNew(m, k, channel, withGroup, withAtt, withDropout, alias)
        self._cLoss = CompressionLossNew()
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, code, logit = self._compressor(image, temp)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, logit)
        return (ssimLoss, l1l2Loss, reg), (restored, code, logit)


class WholePQTwoPass(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._compressor = PQCompressorTwoPass(m, k, channel, withGroup, withAtt, withDropout, alias)
        self._cLoss = CompressionLoss()
        self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, first, **_):
        restored, (quantized, latent), codes, logits = self._compressor(image, temp, first)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, quantized, logits, latent)
        self._movingMean -= 0.9 * (self._movingMean - ssimLoss.mean())
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, codes, quantized, logits, None)


class WholePQRelax(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._compressor = PQRelaxCompressor(m, k, channel, withGroup, withAtt, withDropout, alias)
        self._cLoss = CompressionLossS()

    def forward(self, image, **_):
        restored, qSamples, code, logit = self._compressor(image)

        ssim, reg = self._cLoss(image, restored, qSamples, logit, None)
        storch.add_cost(ssim, "ssim")
        # storch.add_cost(reg, "reg")

        return restored, code, qSamples, logit, None


class WholePQContext(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._compressor = PQContextCompressor(m, k, channel, numLayers)
        self._cLoss = CompressionLoss()

    def forward(self, image, temp, step, **_):
        # maskedImage = self._masking(image)
        restored, codes, logits, predicts, targets = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, None, logits, None)
        contextLoss = self._mLoss(predicts, targets, step)
        return (ssimLoss, l1l2Loss, contextLoss, reg), (restored, codes, predicts, logits, None)


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
