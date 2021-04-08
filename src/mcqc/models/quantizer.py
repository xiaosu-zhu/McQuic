from typing import Callable, List, Optional
from math import sqrt

import storch
from storch.method import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, OneHotCategorical
from cfmUtils.base import Module
from cfmUtils.metrics.pairwise import l2DistanceWithNorm, l2Distance

from mcqc.layers.gumbelSoftmax import GumbelSoftmax
from mcqc.layers.layerGroup import LayerGroup
from mcqc.layers.blocks import ResidualBlock, L2Normalize
from mcqc.models.transformer import Encoder, Decoder
from mcqc.layers.positional import PositionalEncoding2D, PositionalEncoding1D, DePositionalEncoding2D, LearnablePositionalEncoding2D, NPositionalEncoding2D
from mcqc import Consts


class AttentiveQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        cSplitted = cin // len(k)
        for i, numCodewords in enumerate(k):
            setattr(self, f"codebook{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cSplitted))))
        self._codebookAsKey = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._norm = L2Normalize()
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)
        self._gumbel = GumbelSoftmax()

    def getCodebook(self):
        for i in range(len(self._k)):
            yield getattr(self, f"codebook{i}")

    def _attention(self, x, temp, hard):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        softs = list()
        quantizeds = list()
        logits = list()
        for i, _ in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            x = xs[i]
            # [h*w, n, c]
            query = self._xAsQuery[i](x)
            # [k, c]
            key = self._codebookAsKey[i](codewords)
            # [k, c]
            value = self._codebookAsValue[i](codewords)
            # k = self._k[i]
            # [h*w, n, k]
            logit = (query @ key.t()) / self._scaling[i]
            logits.append(logit)
            sample, soft = self._gumbel(logit, temp, hard)
            samples.append(sample)
            softs.append(soft @ value)
            quantizeds.append(sample @ value)
        #      [h*w, n, Cin]
        return torch.cat(quantizeds, -1), samples, torch.cat(softs, -1), logits

    def _attentionEncoder(self, x):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        for i, _ in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            x = xs[i]
            query = self._xAsQuery[i](x)
            key = self._codebookAsKey[i](codewords)
            # [h*w, n, k]
            logit = (query @ key.t()) / self._scaling[i]
            sample = F.gumbel_softmax(logit, 0.1, True)
            # [h*w, n]
            samples.append(sample.argmax(-1))
        return samples

    def _attentionDecoder(self, samples):
        quantizeds = list()
        for i, (k, code) in enumerate(zip(self._k, samples)):
            # n, h, w = code.shape
            codewords = getattr(self, f"codebook{i}")
            value = self._codebookAsValue[i](codewords)
            # [n, h, w, k]
            oneHot = F.one_hot(code, k).float()
            # [n, h, w, c]
            quantizeds.append(oneHot @ value)
        # [n, Cin, h, w]
        return torch.cat(quantizeds, -1).permute(0, 3, 1, 2)

    def encode(self, latents):
        # samples = list()
        zs = list()
        for _, xRaw in enumerate(latents):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h*w, n, c]
            x = encoderIn.reshape(-1, n ,c)
            zs.append(x)
            # [n, h, w]
            samples = self._attentionEncoder(x)
            samples = [s.permute(1, 0).reshape(n, h, w) for s in samples]
        return samples, zs

    def decode(self, codes):
        quantizeds = list()
        # [n, h, w, c]
        quantized = self._attentionDecoder(codes)

        # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
        deTransformed = quantized
        # [n, c, h, w]
        quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, temp, *_):
        quantizeds = list()
        codes = list()
        softQs = list()
        # logits = list()
        for _, (xRaw, k) in enumerate(zip(latents, self._k)):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            """ *************** TODO: NEED DETACH? ******************* """
            # encoderIn = xRaw.detach().permute(2, 3, 0, 1)
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            x = encoderIn.reshape(-1, n ,c)
            # similar to scaled dot-product attention
            # [h*w, N, Cin],    M * [h*w, n, k]
            quantized, samples, softQ, logits = self._attention(x, temp, True)
            # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
            deTransformed = quantized.reshape(h, w, n, c).permute(2, 3, 0, 1)

            # mask = torch.rand_like(xRaw) > coeff
            # mixed = mask * xRaw.detach() + torch.logical_not(mask) * deTransformed
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            softQs.append(softQ.reshape(h, w, n, c).permute(2, 3, 0, 1))
            samples = [s.argmax(-1).permute(1, 0).reshape(n, h, w) for s in samples]
            logits = [l.permute(1, 0, 2).reshape(n, h, w, k) for l in logits]
            # codes.append(samples.argmax(-1).permute(1, 0).reshape(n, h, w))
            # logits.append(logit.permute(1, 0, 2).reshape(n, h, w, k))
        return quantizeds, softQs, codes, logits


class TransformerQuantizer(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        # self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, dropout=rate, activation="gelu"), 3)
        # self._decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, dropout=rate, activation="gelu"), 3)
        cSplitted = cin // len(k)
        for i, numCodewords in enumerate(k):
            setattr(self, f"codebook{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cSplitted))))
            # setattr(self, f"tCodebook{i}", nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(numCodewords, cin))))
            # setattr(self, f"palette{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cin))))
        self._codebookAsKey = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tCodebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tCodebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tXAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)

    def _attention(self, x, temp, hard):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        quantizeds = list()
        logits = list()
        for i, k in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            x = xs[i]
            query = self._xAsQuery[i](x)
            key = self._codebookAsKey[i](codewords)
            value = self._codebookAsValue[i](codewords)
            k = self._k[i]
            # [h*w, n, k]
            logit = (query @ key.t()) / self._scaling[i]
            logits.append(logit)
            sample = F.gumbel_softmax(logit, temp, hard)
            samples.append(sample)
            quantizeds.append(sample @ value)
        #      [h*w, n, Cin]
        return torch.cat(quantizeds, -1), samples, logits

    def _attentionEncoder(self, x):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        for i, k in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            x = xs[i]
            query = self._xAsQuery[i](x)
            key = self._codebookAsKey[i](codewords)
            # [h*w, n, k]
            logit = (query @ key.t()) / self._scaling[i]
            sample = F.gumbel_softmax(logit, 0.1, True)
            # [h*w, n]
            samples.append(sample.argmax(-1))
        return samples

    def _attentionDecoder(self, samples):
        quantizeds = list()
        for i, (k, code) in enumerate(zip(self._k, samples)):
            n, h, w = code.shape
            codewords = getattr(self, f"codebook{i}")
            value = self._codebookAsValue[i](codewords)
            # [n, h, w, k]
            oneHot = F.one_hot(code, k).float()
            # [n, h, w, c]
            quantizeds.append(oneHot @ value)
        # [n, Cin, h, w]
        return torch.cat(quantizeds, -1).permute(0, 3, 1, 2)

    def encode(self, latents):
        # samples = list()
        zs = list()
        for i, xRaw in enumerate(latents):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            if False:
                # [h, w, n, c] -> [h*w, n, c]
                # encoderIn = encoderIn.reshape(-1, n, c)
                encoderIn = self._position(encoderIn).reshape(-1, n, c)
                x = self._encoder(encoderIn)
            else:
                # [h*w, n, c]
                x = encoderIn.reshape(-1, n ,c)
            zs.append(x)
            # [n, h, w]
            samples = self._attentionEncoder(x)
            samples = [s.permute(1, 0).reshape(n, h, w) for s in samples]
        return samples, zs

    def decode(self, codes):
        quantizeds = list()
        # for i, bRaw in enumerate(codes):
            # n, h, w = bRaw.shape
        # [n, h, w, c]
        quantized = self._attentionDecoder(codes)

        if False:
            n, c, h, w = quantized.shape
            posistedQuantized = self._position(quantized.permute(2, 3, 0, 1)).reshape(-1, n, self._c)
            deTransformed = self._decoder(posistedQuantized).reshape(h, w, n, self._c).permute(2, 3, 0, 1)
        else:
            # [h*w, n, c] -> [n, h*w, c] -> [n, h, w, c]
            deTransformed = quantized
        # [n, c, h, w]
        quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, temp, *_):
        quantizeds = list()
        codes = list()
        # logits = list()
        for i, (xRaw, k) in enumerate(zip(latents, self._k)):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            """ *************** TODO: NEED DETACH? ******************* """
            encoderIn = xRaw.detach().permute(2, 3, 0, 1)
            # encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            if False:
                encoderIn = self._position(encoderIn).reshape(-1, n, c)
                # encoderIn = encoderIn.reshape(-1, n, c)
                # [h*w, n, c]
                x = self._encoder(encoderIn)
            else:
                x = encoderIn.reshape(-1, n ,c)
            # similar to scaled dot-product attention
            # [h*w, N, Cin],    M * [h*w, n, k]
            quantized, samples, logits = self._attention(x, temp, True)
            # quantized = x
            if False:
                # [h*w, n, c]
                posistedQuantized = self._position(quantized.reshape(h, w, n, c)).reshape(-1, n, c)
                deTransformed = self._decoder(posistedQuantized).reshape(h, w, n, c).permute(2, 3, 0, 1)
            else:
                # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
                deTransformed = quantized.reshape(h, w, n, c).permute(2, 3, 0, 1)

            # mask = torch.rand_like(xRaw) > coeff
            # mixed = mask * xRaw.detach() + torch.logical_not(mask) * deTransformed
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            samples = [s.argmax(-1).permute(1, 0).reshape(n, h, w) for s in samples]
            logits = [l.permute(1, 0, 2).reshape(n, h, w, k) for l in logits]
            # codes.append(samples.argmax(-1).permute(1, 0).reshape(n, h, w))
            # logits.append(logit.permute(1, 0, 2).reshape(n, h, w, k))
        return quantizeds, codes, logits


class TransformerQuantizerStorch(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._method = RELAX("z", n_samples=1, in_dim=[1024, 2048])
        # self._encoder = Encoder(1, cin, 8, cin, rate)
        # self._decoder = Decoder(1, cin, 8, cin, rate)
        cSplitted = cin // len(k)
        for i, numCodewords in enumerate(k):
            setattr(self, f"codebook{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cSplitted))))
            # setattr(self, f"tCodebook{i}", nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(numCodewords, cin))))
            # setattr(self, f"palette{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cin))))
        self._codebookAsKey = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tCodebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tCodebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for numCodewords in k])
        # self._tXAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        # self._position = NPositionalEncoding2D(cin, 120, 120)

    def _attention(self, x, temp, hard):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        quantizeds = list()
        logits = list()
        for i, k in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            # [n, h*w, c]
            x = xs[i]
            # [n, h*w, c]
            query = self._xAsQuery[i](x)
            # [k, c]
            key = self._codebookAsKey[i](codewords)
            value = self._codebookAsValue[i](codewords)
            k = self._k[i]
            # [n, h*w, k]
            logit = (query @ key.t()) / self._scaling[i]
            logits.append(logit)
            # [n, h*w, k]
            distribution = OneHotCategorical(probs=logit.softmax(-1), validate_args=False)
            sample = self._method(distribution)
            # sample = F.gumbel_softmax(logit, temp, hard)
            samples.append(sample)
            quantizeds.append(sample @ value)
        #      [n, h*w, Cin]
        return torch.cat(quantizeds, -1), samples, logits

    def _attentionEncoder(self, x):
        xs = torch.chunk(x, len(self._k), -1)
        samples = list()
        for i, k in enumerate(self._k):
            codewords = getattr(self, f"codebook{i}")
            x = xs[i]
            query = self._xAsQuery[i](x)
            key = self._codebookAsKey[i](codewords)
            # [n, h*w, k]
            logit = (query @ key.t()) / self._scaling[i]
            # [n, h*w]
            samples.append(logit.argmax(-1))
        return samples

    def _attentionDecoder(self, samples):
        quantizeds = list()
        for i, (k, code) in enumerate(zip(self._k, samples)):
            n, h, w = code.shape
            codewords = getattr(self, f"codebook{i}")
            value = self._codebookAsValue[i](codewords)
            # [n, h, w, k]
            oneHot = F.one_hot(code, k).float()
            # [n, h, w, c]
            quantizeds.append(oneHot @ value)
        # [n, Cin, h, w]
        return torch.cat(quantizeds, -1).permute(0, 3, 1, 2)

    def encode(self, latents, transform):
        # samples = list()
        zs = list()
        for i, xRaw in enumerate(latents):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [n, h, w, c]
            encoderIn = xRaw.permute(0, 2, 3, 1)
            # [n, h*w, c]
            # encoderIn = self._position(encoderIn).reshape(n, -1, c)
            # x = self._encoder(encoderIn)
            x = encoderIn.reshape(n, -1, c)
            zs.append(x)
            # [n, h, w]
            samples = self._attentionEncoder(x)
            samples = [s.permute(1, 0).reshape(n, h, w) for s in samples]
        return samples, zs

    def decode(self, codes, transform):
        quantizeds = list()
        # for i, bRaw in enumerate(codes):
            # n, h, w = bRaw.shape
        # [n, c, h, w]
        quantized = self._attentionDecoder(codes)

        n, c, h, w = quantized.shape
        # posistedQuantized = self._position(quantized.permute(0, 2, 3, 1)).reshape(n, -1, self._c)
        # deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(n, h, w, self._c).permute(0, 3, 1, 2)
        deTransformed = quantized
        # [n, c, h, w]
        quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, temp, transform):
        quantizeds = list()
        codes = list()
        # logits = list()
        for i, (xRaw, k) in enumerate(zip(latents, self._k)):
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [n, h, w, c]
            encoderIn = xRaw.permute(0, 2, 3, 1)
            # [n, h, w, c] -> [n, h*w, c]
            # encoderIn = self._position(encoderIn).reshape(n, -1, c)
            # [n, h*w, c]
            # x = self._encoder(encoderIn)
            x = encoderIn.reshape(n, -1, c)
            # similar to scaled dot-product attention
            # [N, h*w, Cin],    M * [n, h*w, k]
            quantized, samples, logits = self._attention(x, temp, False)
            # [n, h*w, c]
            # posistedQuantized = self._position(quantized.reshape(n, h, w, c)).reshape(n, -1, c)
            # deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(n, h, w, c).permute(0, 3, 1, 2)

            deTransformed = quantized.reshape(n, h, w, c).permute(0, 3, 1, 2)

            # mask = torch.rand_like(xRaw) > coeff
            # mixed = mask * xRaw.detach() + torch.logical_not(mask) * deTransformed
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            samples = [s.argmax(-1).reshape(n, h, w) for s in samples]
            logits = [l.reshape(n, h, w, k) for l in logits]
            # codes.append(samples.argmax(-1).permute(1, 0).reshape(n, h, w))
            # logits.append(logit.permute(1, 0, 2).reshape(n, h, w, k))
        return quantizeds, codes, logits


class TransformerQuantizerRein(nn.Module):
    def __init__(self, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, dim_feedforward=cin, dropout=rate, activation="gelu"), 6)
        self._decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(cin, 8, dim_feedforward=cin, dropout=rate, activation="gelu"), 6)
        for i, numCodewords in enumerate(k):
            setattr(self, f"codebook{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cin))))
            # setattr(self, f"tCodebook{i}", nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(numCodewords, cin))))
            # setattr(self, f"palette{i}", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(numCodewords, cin))))
        self._codebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        # self._tCodebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        # self._tCodebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        # self._tXAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        self._position = PositionalEncoding2D(cin, 120, 120)

    def _attention(self, x, i):
        codewords = getattr(self, f"codebook{i}")
        query = self._xAsQuery[i](x)
        key = self._codebookAsKey[i](codewords)
        value = self._codebookAsValue[i](codewords)
        k = self._k[i]
        # [h*w, n, k]
        logit = (query @ key.t()) / self._scaling[i]
        distribution = Categorical(logits=logit)
        sample = distribution.sample((1, )).squeeze(0)
        oneHot = F.one_hot(sample, k).float()
        return oneHot @ value, oneHot, logit, -distribution.log_prob(sample).sum(axis=0)

    def _negLogP(self, x, code, i):
        codewords = getattr(self, f"codebook{i}")
        query = self._xAsQuery[i](x)
        key = self._codebookAsKey[i](codewords)
        value = self._codebookAsValue[i](codewords)
        # [h*w, n, k]
        logit = (query @ key.t()) / self._scaling[i]
        distribution = Categorical(logits=logit)
        code = code.reshape(code.shape[0], -1).permute(1, 0)
        k = self._k[i]
        oneHot = F.one_hot(code, k).float()
        negLogP = -distribution.log_prob(code).sum(axis=0)
        return oneHot @ value, logit, negLogP

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

    def encode(self, latents):
        samples = list()
        zs = list()
        for i, xRaw in enumerate(latents):
            n, c, h, w = xRaw.shape
            x = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
            # [n, c, h, w] -> [h, w, n, c]
            # encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            # encoderIn = encoderIn.reshape(-1, n, c)
            # encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # x = self._encoder(encoderIn)
            zs.append(x)
            # [n, h, w]
            sample = self._attentionEncoder(x, i).permute(1, 0).reshape(n, h, w)
            samples.append(sample)
        return samples, zs

    def decode(self, codes):
        quantizeds = list()
        for i, bRaw in enumerate(codes):
            n, h, w = bRaw.shape
            # [h*w, n, c]
            quantized = self._attentionDecoder(bRaw.reshape(n, h*w).permute(1, 0), i)
            deTransformed = quantized.reshape(h, w, n, self._c).permute(2, 3, 0, 1)

            # posistedQuantized = self._position(quantized.reshape(h, w, n, self._c)).reshape(-1, n, self._c)
            # deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, self._c).permute(2, 3, 0, 1)
            # [n, c, h, w]
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, codes):
        if codes is not None:
            logits = list()
            negLogPs = list()
            quantizeds = list()
            for i, (xRaw, code) in enumerate(zip(latents, codes)):
                n, c, h, w = xRaw.shape

                x = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
                # [n, c, h, w] -> [h, w, n, c]
                # encoderIn = xRaw.permute(2, 3, 0, 1)
                # [h, w, n, c] -> [h*w, n, c]
                # encoderIn = self._position(encoderIn).reshape(-1, n, c)
                # [h*w, n, c]
                # x = self._encoder(encoderIn)
                # similar to scaled dot-product attention
                # [h*w, N, c]
                quantized, logit, negLogP = self._negLogP(x, code, i)
                # [n, k, h, w]
                logit = logit.permute(1, 2, 0).reshape(n, -1, h, w)
                # # [h*w, n, k]
                # logit = x @ codewords
                # distribution = Categorical(logits=logit)
                # # [h*w, n]
                # sample = distribution.sample((1, )).squeeze(0)#.permute(1, 0).reshape(n, h, w)
                # logits.append(logit.permute(1, 2, 0).reshape(n, k, h, w))
                # [h*w, n] -> [n]
                negLogPs.append(negLogP)
                logits.append(logit)
                quantizeds.append(quantized.permute(1, 2, 0).reshape(n, c, h, w))
            return quantizeds, logits, negLogPs
        quantizeds = list()
        codes = list()
        logits = list()
        negLogPs = list()
        for i, (xRaw, k) in enumerate(zip(latents, self._k)):
            n, c, h, w = xRaw.shape
            x = xRaw.permute(2, 3, 0, 1).reshape(-1, n, c)
            # [n, c, h, w] -> [h, w, n, c]
            # encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            # encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # [h*w, n, c]
            # x = self._encoder(encoderIn)
            # similar to scaled dot-product attention
            # [h*w, N, c]
            quantized, sample, logit, negLogP = self._attention(x, i)

            # posistedQuantized = self._position(quantized.reshape(h, w, n, c)).reshape(-1, n, c)
            # deTransformed = self._decoder(posistedQuantized, posistedQuantized).reshape(h, w, n, c).permute(2, 3, 0, 1)
            # [h*w, n, c] -> [n, c, h*w] -> [n, c, h, w]
            deTransformed = quantized.reshape(h, w, n, c).permute(2, 3, 0, 1)

            # mask = torch.rand_like(xRaw) > coeff
            # mixed = mask * xRaw.detach() + torch.logical_not(mask) * deTransformed
            # [n, c, h, w]
            quantizeds.append(deTransformed)
            codes.append(sample.argmax(-1).permute(1, 0).reshape(n, h, w))
            logits.append(logit.permute(1, 2, 0).reshape(n, k, h, w))
            # [h*w, n] -> [n]
            negLogPs.append(negLogP)
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
        allCodewords = list()
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
