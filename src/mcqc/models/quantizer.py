from typing import Callable, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from cfmUtils.base import Module
from cfmUtils.metrics.pairwise import l2DistanceWithNorm, l2Distance

from mcqc.layers.layerGroup import LayerGroup
from mcqc.layers.gumbelSoftmax import GumbelSoftmax
from mcqc.layers.blocks import ResidualBlock
from mcqc.layers.positional import PositionalEncoding2D, PositionalEncoding1D, DePositionalEncoding2D
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
        # self._squeeze = nn.ModuleList([Transit(numCodewords, cin) for numCodewords in k])
        # self._prob = nn.ModuleList([Transit(cin, numCodewords) for numCodewords in k])
        self._squeeze = nn.ModuleList([Transit(numCodewords, cin, order="last") for numCodewords in k])
        self._prob = nn.ModuleList([Transit(cin, numCodewords, order="first") for numCodewords in k])
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
            # x = self._dePosition(x.reshape(h, w, n, c)).reshape(-1, n, c)
            # x = self._encoder(posisted, codewords[:, None, ...].expand(k, n, c))
            # [h*w, n, k]
            logit = torch.matmul(x, codewords)
            # logit = prob(x, h, w)
            # [n, h, w]
            sample = logit.permute(1, 0, 2).reshape(n, h, w, k).argmax(-1)
            # sample = sample.reshape(n, h, w, k).argmax(-1)
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

    def forward(self, latents, temperature, hard):
        quantizeds = list()
        codes = list()
        logits = list()
        # probability = mixin / (mixin + 1.0)
        # rolloutDistribution = Bernoulli(probs=torch.tensor(probability).to(latents[0].device))
        for xRaw, prob, squeeze, codebook, k in zip(latents, self._prob, self._squeeze, self._codebook, self._k):
            n, c, h, w = xRaw.shape
            # [c, k]
            codewords = codebook.weight
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            # x = self._encoder(posisted)
            x = self._encoder(encoderIn)
            # x = self._dePosition(x.reshape(h, w, n, c)).reshape(-1, n, c)
            # x = encoderIn
            # [h*w, n, k]
            # logit = prob(x, h, w)
            logit = torch.matmul(x, codewords)
            # soft = (logit / temperature).softmax(-1)
            # if hard:
            #     hard = logit.argmax(-1)
            #     hard = F.one_hot(hard, k)
            #     sample = (hard - soft).detach() + soft
            # else:
            #     sample = soft
            sample = F.gumbel_softmax(logit, temperature, hard)
            # sample = logit
            # [h*w, N, c] <- [h*w, N, k] @ [k, C]
            quantized = codebook(sample)

            # quantized = sample

            # normalize
            # quantized /= (k - 0.5) / (2 * k - 2)
            # quantized -= 0.5 / (k - 1)
            # [h*w, n, c]
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
            codes.append(sample.argmax(-1).permute(1, 0).reshape(n, h, w))
            logits.append(logit.reshape(n, h, w, k))
        return quantizeds, codes, logits


class VQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        # self._squeeze = nn.ModuleList([Transit(numCodewords, cin) for numCodewords in k])
        # self._prob = nn.ModuleList([Transit(cin, numCodewords) for numCodewords in k])
        self._squeeze = nn.ModuleList([Transit(cin, cin, order="last") for numCodewords in k])
        self._prob = nn.ModuleList([Transit(cin, cin, order="first") for numCodewords in k])
        # self._encoder = nn.Transformer(cin, )
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, cin, rate), 6, None) #nn.LayerNorm(cin, Consts.Eps))
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, cin, rate), 6, None) #nn.LayerNorm(cin, Consts.Eps))
        self._codebook = nn.ModuleList([nn.Linear(numCodewords, cin, bias=False) for numCodewords in k])
        self._k = k
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)

    def encode(self, latents):
        samples = list()
        for xRaw, codebook, prob, k in zip(latents,  self._codebook, self._prob, self._k):
            n, c, h, w = xRaw.shape
            # [c, k] -> [k, c]
            codewords = codebook.weight.t()
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            posisted = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            # x = self._encoder(posisted)
            x = self._encoder(posisted, codewords[:, None, ...].expand(k, n, c))
            # [h*w, n, k]
            logit = prob(x, h, w)
            # [n, h, w]
            sample = logit.permute(1, 0, 2).reshape(n, h, w, k).argmax(-1)
            # sample = sample.reshape(n, h, w, k).argmax(-1)
            samples.append(sample)
        return samples

    def _oneHotEncode(self, b, k):
        # [n, h, w, k]
        oneHot = F.one_hot(b, k).float()
        # [n, k, h, w]
        return oneHot.permute(0, 3, 1, 2)

    def decode(self, codes):
        quantizeds = list()
        for bRaw, squeeze, k in zip(codes, self._squeeze, self._k):
            n, h, w = bRaw.shape
            # [n, k, h, w], k is the one hot embedding
            bRaw = self._oneHotEncode(bRaw, k)
            # [n, h*w, k] = [n, k, h, w] -> [n, h, w, k] -> [n, h*w, k]
            b = bRaw.permute(0, 2, 3, 1).reshape(n, -1, k)
            # [n, h*w, k] -> [h*w, n, k]
            quantized = b.permute(1, 0, 2)
            quantized /= (k - 0.5) / (2 * k - 2)
            quantized -= 0.5 / (k - 1)
            # [h*w, n, c]
            quantized = squeeze(quantized, h, w)
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = self._decoder(quantized, quantized).permute(1, 2, 0).reshape(n, self._c, h, w)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, temperature, hard, mixin):
        quantizeds = list()
        codes = list()
        logits = list()
        highOrders = list()
        softs = list()
        hards = list()
        probability = mixin / (mixin + 1.0)
        rolloutDistribution = Bernoulli(probs=torch.tensor(probability).to(latents[0].device))
        for xRaw, pre, post, codebook, k in zip(latents, self._prob, self._squeeze, self._codebook, self._k):
            n, c, h, w = xRaw.shape
            # [k, c]
            codewords = codebook.weight
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            posisted = self._position(encoderIn).reshape(-1, n, c)
            encoderIn = encoderIn.reshape(-1, n, c)
            # [h*w, n, c]
            # x = self._encoder(posisted)
            x = self._encoder(posisted)
            # [h*w, n, c]
            preProcess = pre(x, h, w)
            highOrders.append(preProcess.permute(1, 2, 0).reshape(n, c, h, w))
            # [h*w*n, k]
            # similarity = -l2DistanceWithNorm(preProcess.reshape(h*w*n, -1), codewords.t())
            similarity = preProcess @ codewords
            similarity = similarity.reshape(h*w, n, -1)
            logits.append(similarity.permute(1, 0, 2).reshape(n, h, w, -1))

            # sample = F.gumbel_softmax(similarity, temperature, hard)
            # quantized = codebook(sample)

            soft = codebook(torch.softmax(similarity, -1))
            hardCode = similarity.argmax(-1)
            oneHot = nn.functional.one_hot(hardCode, k).float()
            hard = codebook(oneHot)
            # b = (oneHot - soft).detach() + soft
            # quantized = codebook(b)
            quantized = (hard - soft).detach() + soft

            softs.append(soft.permute(1, 2, 0).reshape(n, c, h, w))
            hards.append(hard.permute(1, 2, 0).reshape(n, c, h, w))

            # [h*w, n, c]
            postProcess = post(quantized, h, w)
            # posistedQuantized = self._position(quantized.reshape(h, w, n, c))
            # mixed = (mixin * encoderIn / (mixin + 1)) + (quantized / (mixin + 1))

            mask = rolloutDistribution.sample((h*w, n, 1)).bool()

            mixed = mask * encoderIn.detach() + torch.logical_not(mask) * postProcess

            # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
            deTransformed = self._decoder(mixed, postProcess).permute(1, 2, 0).reshape(n, c, h, w)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            codes.append(hardCode.permute(1, 0).reshape(n, h, w))
            # print(highOrders[-1][0, 0, :4, :4])
            print(hards[-1][0, 0, :4, :4])
            # logits.append(logit.reshape(n, h, w, k))
        return quantizeds, highOrders, softs, hards, codes, logits
