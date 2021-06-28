import math
from typing import List
from math import sqrt

from storch.method import RELAX
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, OneHotCategorical

from mcqc.layers.dropout import PointwiseDropout
from mcqc.models.transformer import Encoder, Transformer
from mcqc.layers.positional import NPositionalEncoding2D


def conv(in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )
def deconv(in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


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
        self._codebookAsKey = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for _ in k])
        # self._tCodebookAsKey = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._codebookAsValue = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for _ in k])
        # self._tCodebookAsValue = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._xAsQuery = nn.ModuleList([nn.Linear(cSplitted, cSplitted) for _ in k])
        # self._tXAsQuery = nn.ModuleList([nn.Linear(cin, cin) for numCodewords in k])
        self._k = k
        self._scaling = [sqrt(kk) for kk in k]
        self._d = float(cin) ** 0.5
        self._c = cin
        # self._position = NPositionalEncoding2D(cin, 120, 120)

    def _attention(self, x: torch.Tensor, temp: float, hard: bool):
        xs = torch.chunk(x, len(self._k), -1)
        samples: List[torch.Tensor] = list()
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


class TransformerQuantizer(nn.Module):
    def __init__(self, layers: int, k: List[int], cin: int, rate: float = 0.1):
        super().__init__()
        k = k[0]
        self._position = NPositionalEncoding2D(cin, 120, 120)
        self._latentTransform = Encoder(layers, cin, 8, cin, rate)
        self._encoder = Transformer(layers, cin, 8, cin, rate)
        setattr(self, "codebook", nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, cin))))
        self._codebookQuery = Encoder(layers, cin, 8, cin, rate)
        self._codebookValue = Encoder(layers, cin, 8, cin, rate)
        self._select = nn.Linear(cin, k)
        self._quantizedTransform = Encoder(layers, cin, 8, cin, rate)
        self._decoder = Transformer(layers, cin, 8, cin, rate)
        self._codebookOut = Encoder(layers, cin, 8, cin, rate)

        # self._sqrtD = math.sqrt(k)

    def encode(self, latents):
        samples = list()
        zs = list()
        for xRaw in latents:
            n, c, h, w = xRaw.shape
            # [1, k, c]
            codebook = getattr(self, "codebook")[None, ...]
            # [n, c, h, w] -> [n, h, w, c]
            encoderIn = xRaw.permute(0, 2, 3, 1)
            # [n, h, w, c] -> [n, h*w, c]
            encoderIn = self._position(encoderIn).reshape(n, -1, c)
            encoderIn = self._latentTransform(encoderIn)
            # [1, k, c]
            codebookQ = self._codebookQuery(codebook)
            # [n, h*w, c]
            x = self._encoder(encoderIn, codebookQ)
            # [n, h*w, k]
            logit = self._select(x)

            # logit = logit - logit.mean(-1, keepdim=True)
            # logit = logit / logit.std(-1, keepdim=True)
            # [n, h, w]
            sample = F.gumbel_softmax(logit, 1.0, True).argmax(-1).reshape(n, h, w)
            zs.append(x)
            samples.append(sample)
        return samples, zs

    def decode(self, codes):
        quantizeds = list()
        for bRaw in codes:
            n, h, w = bRaw.shape
            # [1, k, c]
            codebook = getattr(self, "codebook")[None, ...]
            _, k, c = codebook.shape
            # [1, k, c]
            codewords = self._codebookValue(codebook)
            # [n, h, w, k]
            sample = F.one_hot(bRaw, k).float()
            # [n, h, w, c]
            quantized = sample @ codewords[0, ...]
            # [n, h*w, c]
            posistedQuantized = self._position(quantized).reshape(n, -1, c)
            posistedQuantized = self._quantizedTransform(posistedQuantized)
            # [1, k, c]
            decodedCodes = self._codebookOut(codebook)
            # [n, c, h, w]
            deTransformed = self._decoder(posistedQuantized, decodedCodes).reshape(n, h, w, c).permute(0, 3, 1, 2)
            quantizeds.append(deTransformed)
        return quantizeds

    def forward(self, latents, maskProb, temperature, *_):
        quantizeds = list()
        codes = list()
        logits = list()
        xs = list()
        transformedCodewords = list()
        for xRaw in latents:
            n, c, h, w = xRaw.shape
            # [1, k, c]
            codebook = getattr(self, "codebook")[None, ...]
            transformedCodewords.append(codebook)
            # [n, c, h, w] -> [n, h, w, c]
            encoderIn = xRaw.permute(0, 2, 3, 1)
            # [n, h, w, c] -> [n, h*w, c]
            encoderIn = self._position(encoderIn).reshape(n, -1, c)

            # [n, 1, h*w, h*w]
            contextMask = Bernoulli(probs=torch.ones((encoderIn.shape[1], encoderIn.shape[1]), device=encoderIn.device) * 0.0).sample((n, )).bool()[:, None, ...]

            encoderIn = self._latentTransform(encoderIn, contextMask)
            # [1, k, c]
            codebookQ = self._codebookQuery(codebook)
            # [n, h*w, c]
            x = self._encoder(encoderIn, codebookQ)
            xs.append(x)
            # [n, h*w, k]
            logit = self._select(x)

            # [k]
            bernoulli = Bernoulli(probs=maskProb)
            # [n, h*w, k] (0 or 1 -> choose or not choose)
            randomFalseMask = bernoulli.sample((n, h*w, )).bool()

            maskedLogit = logit.masked_fill(randomFalseMask, -1e9)

            # randomFalseMask *= -1e9
            # maskedLogit = logit + randomFalseMask # + randomTrueMask

            sample = F.gumbel_softmax(maskedLogit, 1.0, True)
            # [1, k, c]
            codewords = self._codebookValue(codebook)
            # [n, h*w, c]
            quantized = sample @ codewords[0, ...]
            # [n, h*w, c]
            posistedQuantized = self._position(quantized.reshape(n, h, w, c)).reshape(n, -1, c)
            posistedQuantized = self._quantizedTransform(posistedQuantized)
            # [1, k, c]
            decodedCodes = self._codebookOut(codebook)
            # [n, c, h, w]
            deTransformed = self._decoder(posistedQuantized, decodedCodes).reshape(n, h, w, c).permute(0, 3, 1, 2)

            # [n, c, h, w]
            quantizeds.append(deTransformed)
            codes.append(sample.argmax(-1).reshape(n, h, w))
            logits.append(logit.reshape(n, h, w, -1))
        return quantizeds, codes, logits, (xs, transformedCodewords)



class AttentiveQuantizer(nn.Module):
    def __init__(self, k: int, cin: int, dropout: bool = True, deterministic: bool = False, additionWeight: bool = True):
        super().__init__()

        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, cin)))

        if additionWeight:
            self._wq = nn.Linear(cin, cin, bias=False)
            self._wk = nn.Linear(cin, cin, bias=False)
            self._wv = nn.Linear(cin, cin, bias=False)
        self._scale = math.sqrt(cin)
        self._additionWeight = additionWeight
        self._deterministic = deterministic
        if dropout:
            self._dropout = PointwiseDropout(0.05)
        else:
            self._dropout = None
        self._randomMask = torch.distributions.Bernoulli(probs=0.95)

    def encode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        # [k, c]
        k = self._codebook

        if self._additionWeight:
            # [n, h, w, c]
            q = self._wq(q)
            # [k, c]
            k = self._wk(k)

        # [n, h, w, k]
        logit = q @ k.permute(1, 0) / self._scale
        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1)

    def _softStraightThrough(self, logit, value):
        k, c = value.shape
        soft = logit.softmax(-1)
        sample = logit.argmax(-1)
        sample = F.one_hot(sample, k).float()
        soft = soft @ value
        hard = sample @ value
        return (hard - soft).detach() + soft, sample

    def decode(self, code):
        k, c = self._codebook.shape
        # [n, h, w, k]
        sample = F.one_hot(code, k).float()
        v = self._codebook
        if self._additionWeight:
            v = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = (sample @ v).permute(0, 3, 1, 2)

        return quantized

    def _gumbelAttention(self, q, k, v, mask, temperature=1.0):
        if self._additionWeight:
            # [n, h, w, c]
            q = self._wq(q)
            # [k, c]
            k = self._wk(k)
            v = self._wv(v)

        # [n, h, w, k]
        logit = (q @ k.permute(1, 0)) / self._scale
        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            logit = logit.masked_fill(mask, -1e9)
        if self._deterministic:
            result, sample = self._softStraightThrough(logit, v)
        else:
            # [n, h, w, k]
            sample = F.gumbel_softmax(logit, temperature, True)
            result = sample @ v
        return result, sample, logit, v

    # def _randomErase(self, x: torch.Tensor):
    #     n, c, h, w = x.shape
    #     # [n, h, w]
    #     mask = Bernoulli(0.15).sample((n, h, w)).bool()
    #     # [n, h, w, c]
    #     x = x.permute(0, 2, 3, 1)
    #     # [?, c]
    #     selectedX = x[mask]
    #     # [?]
    #     mu = selectedX.mean(axis=-1)
    #     # [?]
    #     sigma = selectedX.std(axis=-1)
    #     # [c, ?] -> [?, c]
    #     noise = torch.distributions.Normal(mu, sigma).sample((c, )).permute(1, 0)
    #     x[mask] = noise
    #     return x.permute(0, 3, 1, 2)

    def forward(self, latent, temperature, *_):
        n, _, h, w = latent.shape
        k = self._codebook.shape[0]
        # randomMask = self._randomMask.sample((n, h, w, k)).bool().to(latent.device)
        quantized, sample, logit, wv = self._gumbelAttention(latent.permute(0, 2, 3, 1), self._codebook, self._codebook, None, temperature)
        quantized = quantized.permute(0, 3, 1, 2)
        if self._dropout is not None:
            quantized = self._dropout(quantized)
        # [n, c, h, w], [n, h, w], [n, h, w, k], [k, c]
        return quantized, sample.argmax(-1).byte(), logit, wv
