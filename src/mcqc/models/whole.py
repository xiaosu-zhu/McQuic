import enum
from torch import nn
import torch.nn.functional as F
import torch
import storch

from mcqc.losses.quantization import CompressionLoss, CompressionLossBig, CompressionLossNew, CompressionLossQ, L1L2Loss, QError
from mcqc.models.compressor import AQCompressor, PQCompressor, PQCompressorBig, PQCompressorNew, PQCompressorQ, PQContextCompressor, PQRelaxCompressor, PQCompressorTwoPass, PQCompressor5x5
from mcqc.models.pixelCNN import PixelCNN


class WholePQ(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._compressor = PQCompressor(m, k, channel, withGroup, withAtt, withDropout, alias, ema)
        self._cLoss = CompressionLoss()
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, quantized, latent, trueCodes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, trueCodes, trueCodes, logits, None, None)
        # self._movingMean -= 0.9 * (self._movingMean - ssimLoss.mean())
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, trueCodes, quantized, logits)

class WholePQBig(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super().__init__()
        self._compressor = PQCompressorBig(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = CompressionLossBig(target)
        self._auxLoss = L1L2Loss()
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, allHards, latent, allCodes, allTrues, allLogits, (allFeatures, allQuantizeds), allCodebooks = self._compressor(image, temp, True)

        dLoss = self._cLoss(image, restored)

        # regLoss = list()
        weakCodebookLoss = list()
        weakDiversityLoss = list()
        weakFeatureLoss = list()

        for features, quantizeds, codebooks in zip(allFeatures, allQuantizeds, allCodebooks):
            for codebook in codebooks:
                # [k, k] := [k, c] @ [c, k]
                innerProduct = codebook @ codebook.T
                # orthogonal regularization
                weakCodebookLoss.append(self._auxLoss(innerProduct, torch.eye(innerProduct.shape[0], device=innerProduct.device, dtype=innerProduct.dtype)))
            m = len(features)
            for i in range(m):
                for j in range(i + 1, m):
                    # [n, h, w] := ([n, h, w, c] * [n, h, w, c]).sum(-1)
                    interProduct = (features[i] * features[j]).sum(-1)
                    # feature from different group should be orthogonal
                    weakFeatureLoss.append(2 * self._auxLoss(interProduct, torch.zeros_like(interProduct)))
                intraProduct = (features[i] * features[i]).sum(1)
                # weakDiversityLoss.append(F.mse_loss(quantizeds[i], features[i].detach()))
                weakFeatureLoss.append(self._auxLoss(intraProduct, torch.ones_like(intraProduct)))

        # self._movingMean -= 0.9 * (self._movingMean - ssimLoss.mean())
        # pLoss = self._pLoss(image, restored)
        return dLoss, (sum(weakCodebookLoss), sum(weakFeatureLoss), 0.0), (restored, allTrues, allLogits)

class WholePQPixelCNN(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super().__init__()
        self._levels = len(k)
        self._compressor = PQCompressorBig(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = nn.CrossEntropyLoss()
        self._pixelCNN = nn.ModuleList(PixelCNN(m, ki, channel) for ki in k)
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def test(self, image):
        restored, allCodes, allHards = self._compressor.test(image)
        for i, (pixelCNN, hard, code) in enumerate(zip(self._pixelCNN, allHards, allCodes)):
            logits = pixelCNN(hard)
            correct = logits.argmax(1) == code
            code[correct] = -1
            code += 1

        return restored, allCodes

    def forward(self, image, temp, **_):
        with torch.no_grad():
            allZs, allHards, allCodes, allResiduals = self._compressor.getLatents(image)
        predictLoss = list()
        ratios = list()
        for i, (pixelCNN, hard, code) in enumerate(zip(self._pixelCNN, allHards, allCodes)):
            n, m, c, h, w = hard.shape
            logits = pixelCNN(hard.reshape(n, m * c, h, w))
            dLoss = self._cLoss(logits, code)
            predictLoss.append(dLoss)
            ratios.append((logits.argmax(1) == code).float().mean())
        return sum(predictLoss), sum(ratios) / len(ratios)


class WholePQ5x5(WholePQBig):
    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super(WholePQBig, self).__init__()
        self._compressor = PQCompressor5x5(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = CompressionLossBig(target)
        self._auxLoss = L1L2Loss()
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

class WholePQQ(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._compressor = PQCompressorQ(m, k, channel, withGroup, withAtt, withDropout, alias, ema)
        self._cLoss = CompressionLossQ()
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, quantized, latent, trueCodes, logits = self._compressor(image, temp, True)

        ssimLoss, l1l2Loss, reg = self._cLoss(image, restored, trueCodes, logits)
        # self._movingMean -= 0.9 * (self._movingMean - ssimLoss.mean())
        # pLoss = self._pLoss(image, restored)
        return (ssimLoss, l1l2Loss, reg), (restored, trueCodes, quantized, logits)


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
        return (ssimLoss, l1l2Loss, reg), (restored, trueCodes, quantized, logits, frequencyMaps)


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
