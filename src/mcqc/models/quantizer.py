from types import FrameType
from typing import Union
import math
from storch.method.relax import RELAX

import numpy as np

import torch
from torch import nn
from torch.distributions import categorical
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch import distributed as dist

from mcqc.layers.dropout import PointwiseDropout
from mcqc.layers.stochastic import gumbelRaoMCK, iGumbelSoftmax


class QuantizeBlock(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self._m = m
        self._scale = math.sqrt(k)

    def forward(self, logit: torch.Tensor, temperature):
        n, _, h, w = logit.shape
        logit = logit.reshape(n, self._m, -1, h, w) / self._scale
        return iGumbelSoftmax(logit, temperature, False, dim=2).reshape(n, -1, h, w) if self.training else torch.zeros_like(logit, memory_format=torch.legacy_contiguous_format).scatter_(2, logit.argmax(2, keepdim=True), 1.0).reshape(n, -1, h, w), logit

class AttentiveQuantizer(nn.Module):
    def __init__(self, k: int, cin: int, cout: int, dropout: bool = True, deterministic: bool = False, additionWeight: bool = True, ema: float = 0.8):
        super().__init__()
        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, cout)))
        if ema is not None:
            self._decay = ema
            self.register_buffer("_shadowCodebook", self._codebook.clone().detach())
        else:
            self._decay = None
        if cin != cout and not additionWeight:
            raise AttributeError(f"Can't perform {cin} -> {cout} quantization without additionWeight")
        if additionWeight:
            self._wq = nn.Linear(cin, cout, bias=False)
            self._wk = nn.Linear(cout, cout, bias=False)
            self._wv = nn.Linear(cout, cout, bias=False)
        self._scale = math.sqrt(cin)
        self._additionWeight = additionWeight
        self._deterministic = deterministic
        self._dropout = dropout
        # if dropout:
        #     self._dropout = PointwiseDropout(0.05)
        # else:
        #     self._dropout = None

    @torch.no_grad()
    def EMAUpdate(self):
        if self._decay is not None:
            self._shadowCodebook.data = self._decay * self._shadowCodebook + (1 - self._decay) * self._codebook
            self._codebook.data = self._shadowCodebook

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
        logit = q @ k.permute(1, 0)
        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1)

    def _softStraightThrough(self, logit, temprature, trueCode, binCount, frequency, value):
        n, h, w = frequency.shape
        k, c = value.shape
        soft = (logit / temprature).softmax(-1)
        needMask = (frequency > (float(h * w) / k)).long()
        maxFreq, _ = binCount.max(-1, keepdim=True)
        relaxedFreq = maxFreq + binCount.mean(-1, keepdim=True)
        # reverse frequencies
        # max bin -> meanFreq
        # min bin -> meanFreq + maxbin - minbin
        # [n, k]
        reverseBin = relaxedFreq - binCount
        masked = torch.distributions.Categorical(probs=reverseBin).sample((h, w)).permute(2, 0, 1)
        sample = trueCode * (1 - needMask) + masked * needMask
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

    def _gumbelAttention(self, q, k, v, mask: Union[None, torch.Tensor], temperature: float = 1.0):
        if self._additionWeight:
            # [n, h, w, c]
            q = self._wq(q)
            # [k, c]
            k = self._wk(k)
            v = self._wv(v)

        # [n, h, w, k]
        logit = (q @ k.permute(1, 0)) / self._scale
        n, h, w, k = logit.shape
        with torch.no_grad():
            if self._dropout:
                # [n, h, w, k]
                prob = logit.softmax(-1)
                prob[prob < 1./k] = 0.0
                mask = torch.distributions.Bernoulli(prob).sample().bool()
                # [n, h, w]
                trueCode = prob.argmax(-1)
                # [n, k]
                binCount = torch.zeros((n, k), device=logit.device)
                for i, l in enumerate(trueCode):
                    binCount[i] = torch.bincount(l.flatten(), minlength=k)
                # [n, k] indexed by [n, h, w] -> [n, h, w] frequencies
                ix = torch.arange(n)[:, None, None].expand_as(trueCode)
                frequency = binCount[[ix, trueCode]]
                # [n, h, w]
                dropout = torch.distributions.Bernoulli(frequency / (h * w)).sample().bool()
                # scatter to mask
                # the max of logit has 0.1 probability to be masked
                mask = torch.logical_or(torch.zeros_like(mask, dtype=bool).scatter_(-1, trueCode, dropout[..., None]), mask)
            else:
                # [n, h, w]
                trueCode = logit.argmax(-1)
                # [n, k]
                binCount = torch.zeros((n, k), device=logit.device)
                for i, l in enumerate(trueCode):
                    binCount[i] = torch.bincount(l.flatten(), minlength=k)
                # [n, k] indexed by [n, h, w] -> [n, h, w] frequencies
                ix = torch.arange(n)[:, None, None].expand_as(trueCode)
                frequency = binCount[[ix, trueCode]]
                # frequency = torch.ones_like(logit) * logit.shape[0] / logit.numel()
        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            maskedLogit = logit.masked_fill(mask, -1e9)
        else:
            maskedLogit = logit
        if self._deterministic:
            result, sample = self._softStraightThrough(maskedLogit, temperature, trueCode, binCount, frequency, v)
        else:
            # [n, h, w, k]
            # sample = iGumbelSoftmax(maskedLogit, temperature, True)
            # sample = gumbelRaoMCK(maskedLogit, temperature, 32)
            # sample = torch.distributions.OneHotCategoricalStraightThrough(logits=maskedLogit).rsample(())
            sample = F.gumbel_softmax(maskedLogit, temperature, True)
            result = sample @ v
        return result, sample, logit, (trueCode, frequency, binCount)

    def _argmax(self, q, k, v):
        if self._additionWeight:
            q = self._wq(q)
            k = self._wk(k)
            v = self._wv(v)
        logit = (q @ k.permute(1, 0))
        sample = F.one_hot(logit.argmax(-1), logit.shape[-1]).float()
        result = sample @ v
        frequency = torch.ones_like(logit) * logit.shape[0] / logit.numel()
        return result, sample, logit, frequency, None

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

    def forward(self, latent, temperature):
        n, _, h, w = latent.shape
        k = self._codebook.shape[0]
        # randomMask = self._randomMask.sample((n, h, w, k)).bool().to(latent.device)
        if temperature >= 0:
            quantized, sample, logit, (trueCode, frequency, binCount) = self._gumbelAttention(latent.permute(0, 2, 3, 1), self._codebook, self._codebook, None, temperature)
        else:
            quantized, sample, logit, (trueCode, frequency, binCount) = self._argmax(latent.permute(0, 2, 3, 1), self._codebook, self._codebook)
        quantized = quantized.permute(0, 3, 1, 2)
        # if self._dropout is not None:
        #     quantized = self._dropout(quantized)
        # [n, c, h, w], [n, h, w], [n, h, w, k], [k, c]
        return quantized, sample.argmax(-1), logit, (trueCode, frequency, binCount)


class Quantizer(nn.Module):
    def __init__(self, m: int, k: int, d: int, dropout: bool = True, deterministic: bool = False, additionWeight: bool = True):
        super().__init__()
        self._m = m
        d = d // m
        self._codebook = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, k, d)))
        if additionWeight:
            self._wq = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
            self._wk = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
            self._wv = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
        self._scale = math.sqrt(d)
        self._additionWeight = additionWeight
        self._deterministic = deterministic
        if dropout:
            self._dropout = PointwiseDropout(0.05)
        else:
            self._dropout = None

    def encode(self, latent):
        n, _, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        # [n, h, w, m, d], [m, d, d] -> [n, h, w, m, d]
        # x @ w.t()
        q = torch.einsum("nhwmd,mcd->nhwmc", q, self._wq)
        # [m, k, d], [m, d, d] -> [m, k, d]
        k = torch.einsum("mkd,mcd->mkc", self._codebook, self._wk)
        # [n, h, w, m]
        code = torch.einsum("nhwmd,mkd->nhwmk", q, k).argmax(-1).byte()
        return code

    def decode(self, codes):
        n, h, w, m = codes.shape
        k = self._codebook.shape[1]
        # [n, h, w, m, k]
        oneHot = F.one_hot(codes.long(), k).float()
        # [m, k, d], [m, d, d] -> [m, k, d]
        v = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, c, h, w]
        return torch.einsum("nhwmk,mkc->nhwmc", oneHot, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, latent, temperature, first):
        # [n, h, w, m]
        n, _, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        # [n, h, w, m, d], [m, d, d] -> [n, h, w, m, d]
        # x @ w.t()
        q = torch.einsum("nhwmd,mcd->nhwmc", q, self._wq)
        # [m, k, d], [m, d, d] -> [m, k, d]
        k = torch.einsum("mkd,mcd->mkc", self._codebook, self._wk)
        # [m, k, d], [m, d, d] -> [m, k, d]
        v = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, h, w, m, k]
        logit = torch.einsum("nhwmd,mkd->nhwmk", q, k)
        if first:
            hard = iGumbelSoftmax((logit / self._scale), temperature, False)
        else:
            # [n, h, w, m, k]
            hard = torch.distributions.OneHotCategorical(logits=logit).sample(())
        quantized = torch.einsum("nhwmk,mkc->nhwmc", hard, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)
        # [n, c, h, w], [n, h, w, m], [n, h, w, m, k]
        return quantized, logit.argmax(-1).byte(), logit


class RelaxQuantizer(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        d = d // m
        self._k = k
        self._codebook = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, k, d)))
        self._wq = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
        self._wk = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
        self._wv = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(m, d, d)))
        self._method = RELAX("z", in_dim=[k], rebar=True)

    def encode(self, latent):
        n, _, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        # [n, h, w, m, d], [m, d, d] -> [n, h, w, m, d]
        # x @ w.t()
        q = torch.einsum("nhwmd,mcd->nhwmc", q, self._wq)
        # [m, k, d], [m, d, d] -> [m, k, d]
        k = torch.einsum("mkd,mcd->mkc", self._codebook, self._wk)
        # [n, h, w, m]
        code = torch.einsum("nhwmd,mkd->nhwmk", q, k).argmax(-1).byte()
        return code

    def decode(self, codes):
        n, h, w, m = codes.shape
        k = self._codebook.shape[1]
        # [n, h, w, m, k]
        oneHot = F.one_hot(codes.long(), k).float()
        # [m, k, d], [m, d, d] -> [m, k, d]
        v = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, c, h, w]
        return torch.einsum("nhwmk,mkd->nhwmd", oneHot, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, latent):
        # [n, c, h, w]
        n, c, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        # [n, h, w, m, d], [m, d, d] -> [n, h, w, m, d]
        # x @ w.t()
        q = torch.einsum("nhwmd,mcd->nhwmc", q, self._wq)
        # [m, k, d], [m, d, d] -> [m, k, d]
        k = torch.einsum("mkd,mcd->mkc", self._codebook, self._wk)
        # [m, k, d], [m, d, d] -> [m, k, d]
        v = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, h, w, m, k]
        logit = torch.einsum("nhwmd,mkd->nhwmk", q, k)
        varPosterior = Categorical(logits=logit)
        sample = self._method(varPosterior)
        quantized = torch.einsum("znhwmk,mkd->znhwmd", sample, v).reshape(-1, n, h, w, c).permute(0, 1, 4, 2, 3)
        return quantized, logit.argmax(-1).byte(), logit
