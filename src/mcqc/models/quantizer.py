from typing import Callable, List
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from cfmUtils.base import Module
from cfmUtils.metrics.pairwise import l2DistanceWithNorm, l2Distance

from mcqc.layers.layerGroup import LayerGroup
from mcqc.layers.gumbelSoftmax import GumbelSoftmax
from mcqc.layers.blocks import ResidualBlock
from mcqc.layers.positional import PositionalEncoding2D, PositionalEncoding1D, DePositionalEncoding2D, LearnablePositionalEncoding2D
from mcqc import Consts

class _resBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1, activationFn: Callable = None):
        super().__init__()
        self._layers = nn.Sequential(LayerGroup(d, d, rate, activationFn), LayerGroup(d, 2 * d, rate, activationFn), LayerGroup(2 * d, d, rate, activationFn))

    def forward(self, x):
        return x + self._layers(x)


class _incResBlock(nn.Module):
    def __init__(self, d: int, rate: float = 0.1, activationFn: Callable = None):
        super().__init__()
        self._fc1 = LayerGroup(d, d // 2, rate, activationFn)
        self._fc2 = LayerGroup(d // 2, d // 4, rate, activationFn)
        self._fc3 = LayerGroup(d // 4, d // 2, rate, activationFn)
        self._fc4 = LayerGroup(d // 2, d, rate, activationFn)
        self._res1 = _resBlock(d // 4, rate, activationFn)
        self._res2 = _resBlock(d // 2, rate, activationFn)
        self._res3 = _resBlock(d, rate, activationFn)

    def forward(self, x):
        # d // 2
        x1 = self._fc1(x)
        # d // 4
        x2 = self._fc2(x1)
        # d // 2
        x1 = self._res2(x1)
        # d // 4
        x2 = self._res1(x2)
        # d
        x3 = self._res3(x)
        return x + x3 + self._fc4(self._fc3(x2) + x1)


class Quantizer(nn.Module):
    def __init__(self, k: int, cin: int, rate: float = 0.1):
        super().__init__()
        self._net = nn.Transformer(cin, 8, 1, 1, cin)
        self._net = nn.Linear(cin, k)
        self._codebook = nn.Parameter(torch.randn(k, cin))

    def forward(self, x, temperature, hard):
        x = x.permute(0, 2, 3, 1)
        # [N, h, w, k]
        logits = self._net(x)
        samples = F.gumbel_softmax(logits, temperature, hard)
        # [N, h, w, C] <- [N, h, w, k] @ [k, C]
        quantized = samples @ self._codebook
        return quantized.permute(0, 3, 1, 2), samples, logits.permute(0, 3, 1, 2)


class MultiCodebookQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._net = nn.ModuleList([nn.Linear(cin, numCodewords) for numCodewords in k])
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        samples = list()
        logits = list()
        for x, net, codebook in zip(latents, self._net, self._codebook):
            x = x.permute(0, 2, 3, 1)
            # [N, h, w, k]
            logit = net(x)
            sample = F.gumbel_softmax(logit * self._d, temperature, hard)
            # [N, h, w, C] <- [N, h, w, k] @ [k, C]
            quantized = codebook(sample)

            quantizeds.append(quantized.permute(0, 3, 1, 2))
            samples.append(sample)
            logits.append(logit.permute(0, 3, 1, 2))

        return quantizeds, samples, logits


class TransformerQuantizerBak(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._squeeze = nn.ModuleList([nn.Linear(numCodewords, cin) for numCodewords in k])
        self._prob = nn.ModuleList([nn.Linear(cin, numCodewords) for numCodewords in k])
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, cin, rate), 6, nn.LayerNorm(cin, 1e-6))
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, cin, rate), 6, nn.LayerNorm(cin, 1e-6))
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding1D(cin)

    def encode(self, latents):
        samples = list()
        for xRaw, net, k in zip(latents, self._prob, self._k):
            n, c, h, w = xRaw.shape
            # [h*w, n, c] = [n, c, h, w] -> [h, w, n, c] -> [h*w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
            # [h*w, n, c]
            x = self._encoder(encoderIn)
            # [n, k, h, w] = [h*w, n, k] -> [n, k, h*w] -> [n, k, h, w]
            logit = net(x).permute(1, 2, 0).reshape(n, k, h, w)
            # [n, h, w]
            sample = logit.argmax(1)
            samples.append(sample)

        return samples

    def _oneHotEncode(self, b, k):
        # [n, h, w, k]
        oneHot = F.one_hot(b, k).float()
        # [n, k, h, w]
        return oneHot.permute(0, 3, 1, 2)

    def decode(self, codes):
        quantizeds = list()
        for bRaw, codebook, k in zip(codes, self._codebook, self._k):
            n, h, w = bRaw.shape
            # [n, k, h, w], k is the one hot embedding
            bRaw = self._oneHotEncode(bRaw, k)
            # [n, h*w, k] = [n, k, h, w] -> [n, h, w, k] -> [n, h*w, k]
            b = bRaw.permute(0, 2, 3, 1).reshape(n, -1, k)
            # [N, h*w, c] <- [N, h*w, k] @ [k, C]
            quantized = codebook(b)
            # [n, h*w, c] -> [h*w, n, c]
            quantized = quantized.permute(1, 0, 2)
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(quantized, quantized).permute(1, 0, 2).reshape(n, h, w, self._c)
            # [n, c, h, w]
            quantizeds.append(deTransformed.permute(0, 3, 1, 2))
        return quantizeds

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        samples = list()
        logits = list()
        targets = list()
        for xRaw, prob, squeeze, codebook, k in zip(latents, self._prob, self._squeeze, self._codebook, self._k):
            targets.append(xRaw)
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c] -> [h*w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
            encoderIn = self._position(encoderIn)
            # [h*w, n, c] @ [c, k] -> [h*w, n, k]
            # encoderIn = torch.matmul(encoderIn, codebook.weight)
            # encoderIn = squeeze(encoderIn)
            x = self._encoder(encoderIn)
            # [h*w, n, k] -> [n, h*w, k]
            logit = prob(x).permute(1, 0, 2)
            sample = F.gumbel_softmax(logit * self._d, 1.0, hard)
            # [N, h*w, c] <- [N, h*w, k] @ [k, C]
            quantized = codebook(sample)
            # [n, h*w, c] -> [h*w, n, c]
            quantized = quantized.permute(1, 0, 2)
            # mixin = temperature / 100.0
            # mixed = (temperature * encoderIn / (temperature + 1)) + (quantized / (temperature + 1))

            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(self._position(quantized), quantized).permute(1, 0, 2).reshape(n, h, w, c)
            # [n, c, h, w]
            quantizeds.append(deTransformed.permute(0, 3, 1, 2))
            samples.append(sample)
            logits.append(logit.reshape(n, h, w, k).permute(0, 3, 1, 2))

        return quantizeds, targets, samples, logits

class Transit(nn.Module):
    def __init__(self, cin, cout, order="first"):
        super().__init__()
        self._linear = nn.Linear(cin, cout)
        # if order == "first":
        #     self._preLinear = nn.Linear(cin, cin)
        #     self._preActivation = nn.LeakyReLU(inplace=True)
        #     self._block = ResidualBlock(cin, cin)
        #     self._afterActivation = nn.LeakyReLU(inplace=True)
        #     self._afterLinear = nn.Linear(cin, cout)
        # else:
        #     self._preLinear = nn.Linear(cin, cout)
        #     self._preActivation = nn.LeakyReLU(inplace=True)
        #     self._block = ResidualBlock(cout, cout)
        #     self._afterActivation = nn.LeakyReLU(inplace=True)
        #     self._afterLinear = nn.Linear(cout, cout)

    def forward(self, x, h, w):
        return self._linear(x)
        # hw, n, _ = x.shape
        # x = x.reshape(h, w, n, -1)
        # x = self._preActivation(self._preLinear(x)).permute(2, 3, 0, 1)
        # x = self._block(x).permute(2, 3, 0, 1)
        # x = self._afterActivation(self._afterLinear(x))
        # return x.reshape(hw, n, -1)


class TransformerQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, dropout=rate, activation="gelu"), 3)
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, dropout=rate, activation="gelu"), 3)
        for i, numCodewords in enumerate(k):
            setattr(self, f"codebook{i}", nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(numCodewords, cin))))
            # setattr(self, f"palette{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cin))))
        self._codebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)

    def _attention(self, x, i, evalMode):
        codewords = getattr(self, f"codebook{i}")
        query = self._xAsQuery[i](x)
        key = self._codebookAsKey[i](codewords)
        value = self._codebookAsValue[i](codewords)
        k = self._k[i]
        # [h*w, n, k]
        logit = (query @ key.t()) / self._scaling[i]
        if evalMode:
            sample = logit.argmax(-1)
            # [h*w, n, k]
            oneHot = F.one_hot(sample, k).float()
            # [h*w, n, c]
            return oneHot @ value
        sample = F.gumbel_softmax(logit, 1.0, True)
        return sample @ value, sample, logit

    def _attentionEncoder(self, x, i):
        codewords = getattr(self, f"codebook{i}")
        query = self._xAsQuery[i](x)
        key = self._codebookAsKey[i](codewords)
        # [h*w, n, k]
        logit = (query @ key.t()) / self._scaling[i]
        # [h*w, n]
        return logit.argmax(-1)

    def _attentionDecoder(self, code, i):
        codewords = getattr(self, f"codebook{i}")
        value = self._codebookAsValue[i](codewords)
        k = self._k[i]
        # [h*w, n, k]
        oneHot = F.one_hot(code, k).float()
        # [h*w, n, c]
        return oneHot @ value

    def encode(self, latents, transform):
        samples = list()
        zs = list()
        for i, xRaw in enumerate(latents):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            if transform:
                # [h, w, n, c] -> [h*w, n, c]
                encoderIn = self._position(encoderIn).reshape(-1, n, c)
                x = self._encoder(encoderIn)
            else:
                # [h*w, n, c]
                x = encoderIn.reshape(-1, n ,c)
            zs.append(x)
            # [n, h, w]
            sample = self._attentionEncoder(x, i).permute(1, 0).reshape(n, h, w)
            samples.append(sample)
        return samples, zs

    def decode(self, codes, transform):
        quantizeds = list()
        for i, bRaw in enumerate(codes):
            n, h, w = bRaw.shape
            # [h*w, n, c]
            quantized = self._attentionDecoder(bRaw.reshape(n, h*w).permute(1, 0), i)

            if transform:
                posistedQuantized = self._position(quantized.reshape(h, w, n, self._c)).reshape(-1, n, self._c)
                deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, self._c).permute(2, 3, 0, 1)
            else:
                # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
                deTransformed = quantized.reshape(h, w, n, self._c).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, transform):
        quantizeds = list()
        codes = list()
        logits = list()
        for i, (xRaw, k) in enumerate(zip(latents, self._k)):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            if transform:
                encoderIn = self._position(encoderIn).reshape(-1, n, c)
                # [h*w, n, c]
                x = self._encoder(encoderIn)
            else:
                x = encoderIn.reshape(-1, n ,c)
            # similar to scaled dot-product attention
            # [h*w, N, c]
            quantized, sample, logit = self._attention(x, i, False)
            if transform:
                # [h*w, n, c]
                posistedQuantized = self._position(quantized.reshape(h, w, n, c)).reshape(-1, n, c)
                deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, c).permute(2, 3, 0, 1)
            else:
                # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
                deTransformed = quantized.reshape(h, w, n, c).permute(2, 3, 0, 1)

            # [n, c, h, w]
            quantizeds.append(deTransformed)
            codes.append(sample.argmax(-1).permute(1, 0).reshape(n, h, w))
            logits.append(logit.permute(1, 0, 2).reshape(n, h, w, k))
        return quantizeds, codes, logits


class TransformerQuantizerRein(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, cin, rate, "gelu"), 12, None)
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, cin, rate, "gelu"), 12, None)
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)
        self._dePosition = DePositionalEncoding2D(cin, 120, 120)

    def encode(self, latents):
        samples = list()
        zs = list()
        for xRaw, codebook, k in zip(latents, self._codebook, self._k):
            n, c, h, w = xRaw.shape
            # [c, k] -> [k, c]
            codewords = codebook.weight
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h, w, n, c] -> [h*w, n, c]
            # posisted = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            x = self._encoder(encoderIn)
            zs.append(x)
            # x = self._dePosition(x.reshape(h, w, n, c)).reshape(-1, n, c)
            # x = self._encoder(posisted, codewords[:, None, ...].expand(k, n, c))
            # [h*w, n, k]
            logit = torch.matmul(x, codewords)
            # logit = prob(x, h, w)
            # [n, h, w]
            sample = logit.permute(1, 0, 2).reshape(n, h, w, k).argmax(-1)
            # sample = sample.reshape(n, h, w, k).argmax(-1)
            samples.append(sample)
        return samples, zs

    def _oneHotEncode(self, b, k):
        # [n, h, w, k]
        return F.one_hot(b, k).float()

    def decode(self, codes):
        quantizeds = list()
        for bRaw, codebook, k in zip(codes, self._codebook, self._k):
            n, h, w = bRaw.shape
            # [n, k, h, w], k is the one hot embedding
            bRaw = self._oneHotEncode(bRaw, k)
            # [n, h*w, k] = [n, h, w, k] -> [n, h*w, k]
            b = bRaw.reshape(n, -1, k)
            # [n, h*w, k] -> [h*w, n, k]
            quantized = b.permute(1, 0, 2)
            # quantized /= (k - 0.5) / (2 * k - 2)
            # quantized -= 0.5 / (k - 1)
            # [h*w, n, c]
            quantized = codebook(quantized)

            posistedQuantized = self._position(quantized.reshape(h, w, n, self._c)).reshape(-1, n, self._c)
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, self._c).permute(2, 3, 0, 1)
            # deTransformed = quantized.permute(1, 2, 0).reshape(n, c, h, w)
            # deTransformed = self._dePosition(deTransformed.reshape(h, w, n, self._c)).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, codes=None):
        if codes is not None:
            logits = list()
            negLogPs = list()
            for xRaw, codebook, k in zip(latents, self._codebook, self._k):
                n, c, h, w = xRaw.shape
                # [c, k]
                codewords = codebook.weight
                # [n, c, h, w] -> [h, w, n, c]
                encoderIn = xRaw.permute(2, 3, 0, 1)
                # [h, w, n, c] -> [h*w, n, c]
                encoderIn = self._position(encoderIn).reshape(-1, n, c)
                # [h*w, n, c]
                x = self._encoder(encoderIn)
                # [h*w, n, k]
                logit = x @ codewords
                distribution = Categorical(logits=logit)
                # [h*w, n]
                sample = distribution.sample((1, )).squeeze(0)#.permute(1, 0).reshape(n, h, w)
                logits.append(logit.permute(1, 2, 0).reshape(n, k, h, w))
                # [h*w, n] -> [n]
                negLogPs.append(-distribution.log_prob(sample).sum(axis=(0)))
            return logits, negLogPs
        quantizeds = list()
        codes = list()
        logits = list()
        negLogPs = list()
        for xRaw, codebook, k in zip(latents, self._codebook, self._k):
            n, c, h, w = xRaw.shape
            # [c, k]
            codewords = codebook.weight
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            x = self._encoder(encoderIn)
            # [h*w, n, k]
            logit = x @ codewords
            distribution = Categorical(logits=logit)
            # [h*w, n]
            sample = distribution.sample((1, )).squeeze(0)#.permute(1, 0).reshape(n, h, w)
            # [h*w, n, k]
            oneHot = F.one_hot(sample, k).float()
            # oneHot = self._oneHotEncode(sample, k).reshape(n, -1, k).permute(1, 0, 2)
            # [h*w, N, c] <- [h*w, N, k] @ [k, C]
            quantized = codebook(oneHot)
            posistedQuantized = self._position(quantized.reshape(h, w, n, c)).reshape(-1, n, c)

            # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
            deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, c).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            codes.append(sample.permute(1, 0).reshape(n, h, w))
            logits.append(logit.permute(1, 2, 0).reshape(n, k, h, w))
            # [h*w, n] -> [n]
            negLogPs.append(-distribution.log_prob(sample).sum(axis=(0)))
        return quantizeds, codes, logits, negLogPs


class VQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        # self._squeeze = nn.ModuleList([Transit(numCodewords, cin) for numCodewords in k])
        # self._prob = nn.ModuleList([Transit(cin, numCodewords) for numCodewords in k])
        # self._squeeze = nn.ModuleList([Transit(numCodewords, cin, order="last") for numCodewords in k])
        # self._prob = nn.ModuleList([Transit(cin, numCodewords, order="first") for numCodewords in k])
        # self._encoder = nn.Transformer(cin, )
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, cin, rate, "gelu"), 12, None)
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, cin, rate, "gelu"), 12, None)
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)
        self._dePosition = DePositionalEncoding2D(cin, 120, 120)

    def encode(self, latents):
        samples = list()
        zs = list()
        for xRaw, codebook, k in zip(latents, self._codebook, self._k):
            n, c, h, w = xRaw.shape
            # [c, k]
            codewords = codebook.weight
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h, w, n, c] -> [h*w, n, c]
            # posisted = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            x = self._encoder(encoderIn)
            zs.append(x)
            # x^2 - 2xy + y^2
            # [h*w, n, k]
            d = (x ** 2).sum(-1, keepdim=True) +  (codewords.t() ** 2).sum(-1) - 2 * (x @ codewords)

            # find closest encodings
            # [n, h, w]
            sample = d.argmin(-1).permute(1, 0).reshape(n, h, w)
            # sample = sample.reshape(n, h, w, k).argmax(-1)
            samples.append(sample)
        return samples, zs

    def _oneHotEncode(self, b, k):
        # [n, h, w, k]
        oneHot = F.one_hot(b, k).float()
        # [n, k, h, w]
        return oneHot.permute(0, 3, 1, 2)

    def decode(self, codes):
        quantizeds = list()
        for bRaw, codebook, k in zip(codes, self._codebook, self._k):
            # [c, k]
            codewords = codebook.weight
            n, h, w = bRaw.shape
            # [n, k, h, w], k is the one hot embedding
            bRaw = self._oneHotEncode(bRaw, k)
            # [n, h*w, k] = [n, k, h, w] -> [n, h, w, k] -> [n, h*w, k]
            b = bRaw.permute(0, 2, 3, 1).reshape(n, -1, k)
            # [n, h*w, k] -> [h*w, n, k]
            quantized = b.permute(1, 0, 2)
            # quantized /= (k - 0.5) / (2 * k - 2)
            # quantized -= 0.5 / (k - 1)
            # [h*w, n, c]
            quantized = quantized @ codewords.t()

            posistedQuantized = self._position(quantized.reshape(h, w, n, self._c)).reshape(-1, n, self._c)
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, self._c).permute(2, 3, 0, 1)
            # deTransformed = quantized.permute(1, 2, 0).reshape(n, c, h, w)
            # deTransformed = self._dePosition(deTransformed.reshape(h, w, n, self._c)).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        zq = list()
        zs = list()
        codes = list()
        softs = list()
        # logits = list()
        allCodewords = list()
        # probability = mixin / (mixin + 1.0)
        # rolloutDistribution = Bernoulli(probs=torch.tensor(probability).to(latents[0].device))
        for xRaw, codebook, k in zip(latents, self._codebook, self._k):
            # [c, k]
            codewords = codebook.weight
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            # x = self._encoder(posisted)
            x = self._encoder(encoderIn)

            zs.append(x)

            # x^2 - 2xy + y^2
            # [h*w, n, k]
            d = (x ** 2).sum(-1, keepdim=True) +  (codewords.t() ** 2).sum(-1) - 2 * (x @ codewords)

            # find closest encodings
            # [h*w, n]
            nearest = d.argmin(-1)
            # [h*w, n, k]
            oneHot = F.one_hot(nearest, num_classes=k)
            quantized = oneHot.float() @ codewords.t()
            zq.append(quantized)
            quantized = x # (quantized - x).detach() + x

            maxi, _ = d.max(-1, keepdim=True)
            norm = d / maxi
            soft = norm.softmax(-1)
            softs.append(soft @ codewords.t())
            # logit = norm
            # sample = F.gumbel_softmax(logit, temperature, hard)
            # quantized = codebook(sample)
            # zq.append(quantized)
            # quantized += torch.randn_like(quantized)
            # quantized = sample

            # normalize
            # quantized /= (k - 0.5) / (2 * k - 2)
            # [h*w, n, c]
            # quantized -= 0.5 / (k - 1)
            # quantized = squeeze(sample, h, w)
            posistedQuantized = self._position(quantized.reshape(h, w, n, c)).reshape(-1, n, c)

            # mixed = (mixin * encoderIn / (mixin + 1)) + (quantized / (mixin + 1))

            # mask = rolloutDistribution.sample((h*w, n, 1)).bool()

            # mixed = mask * encoderIn.detach() + torch.logical_not(mask) * quantized
            # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
            deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, c).permute(2, 3, 0, 1)
            # deTransformed = quantized.permute(1, 2, 0).reshape(n, c, h, w)
            # deTransformed = self._dePosition(deTransformed.reshape(h, w, n, c)).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            # codes.append(sample.argmax(-1).permute(1, 0).reshape(n, h, w))
            codes.append(nearest.permute(1, 0).reshape(n, h, w))
            # logits.append(logit.reshape(n, h, w, k))
            allCodewords.append(codewords.t())
        return quantizeds, codes, (zs, zq, softs), allCodewords
