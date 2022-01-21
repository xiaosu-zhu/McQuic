import math
from torch import nn
import torch.nn.functional as F
import torch
import storch

from mcquic.loss.quantization import CodebookSpreading, CompressionLoss, CompressionLossBig, CompressionLossNew, CompressionLossQ, L1L2Loss, L2Regularization, MeanAligning, QError, Regularization
from mcquic.models.compressor import PQCompressorBig, PQCompressor5x5
from mcquic.models.deprecated.pixelCNN import PixelCNN


class WholePQBig(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, target, alias, ema):
        super().__init__()
        self._k = k
        self._compressor = PQCompressorBig(m, k, channel, withGroup, withAtt, False, alias, ema)
        self._cLoss = CompressionLossBig(target)
        # self._auxLoss = L1L2Loss()
        # self._alignLoss = MeanAligning()
        # self._klDivergence = Regularization()
        self._spreadLoss = CodebookSpreading()
        # self._l2Reg = L2Regularization()
        # self.register_buffer("_movingMean", torch.zeros([1]))
        # self._pLoss = LPIPS(net_type='vgg', version='0.1')

    def forward(self, image, temp, **_):
        restored, allHards, latent, allCodes, allTrues, allLogits, (allFeatures, allQuantizeds), allCodebooks = self._compressor(image, temp, True)

        dLoss = self._cLoss(image, restored)

        # regLoss = list()
        # [N, M, H, W, K]
        # for logits in allLogits:
        #     regLoss.append(self._klDivergence(logits))

        # weakFeatureLoss = list()
        weakCodebookLoss = list()

        for raws, codes, codebooks, k, logits in zip(allFeatures, allCodes, allCodebooks, self._k, allLogits):
            for raw, code, codebook, logit in zip(raws, codes, codebooks, logits):
                # weakFeatureLoss.append(self._alignLoss(raw, F.one_hot(code, k).float(), codebook))
                weakCodebookLoss.append(self._spreadLoss(codebook, temp))
                # weakCodebookLoss.append(self._l2Reg(raw, -1))

        return dLoss, (sum(weakCodebookLoss), 0.0, 0.0), (restored, allTrues, allLogits)

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
