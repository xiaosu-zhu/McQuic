from typing import Union
import math
from storch.method.relax import RELAX

import torch
from torch import nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from mcqc.layers.dropout import PointwiseDropout
from mcqc.layers.stochastic import DiscreteReparam, iGumbelSoftmax

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

    def _gumbelAttention(self, q, k, v, mask: Union[None, torch.Tensor], temperature: float = 1.0):
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
            sample = iGumbelSoftmax(logit, temperature, True)
            result = sample @ v
        return result, sample, logit, v

    def _argmax(self, q, k, v):
        if self._additionWeight:
            q = self._wq(q)
            k = self._wk(k)
            v = self._wv(v)
        logit = (q @ k.permute(1, 0))
        sample = F.one_hot(logit.argmax(-1), logit.shape[-1]).float()
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

    def forward(self, latent, temperature):
        n, _, h, w = latent.shape
        k = self._codebook.shape[0]
        # randomMask = self._randomMask.sample((n, h, w, k)).bool().to(latent.device)
        if temperature >= 0:
            quantized, sample, logit, wv = self._gumbelAttention(latent.permute(0, 2, 3, 1), self._codebook, self._codebook, None, temperature)
        else:
            quantized, sample, logit, wv = self._argmax(latent.permute(0, 2, 3, 1), self._codebook, self._codebook)
        quantized = quantized.permute(0, 3, 1, 2)
        if self._dropout is not None:
            quantized = self._dropout(quantized)
        # [n, c, h, w], [n, h, w], [n, h, w, k], [k, c]
        return quantized, sample.argmax(-1).byte(), logit, wv


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
            self._dropout = PointwiseDropout(0.1)
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
        k = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, c, h, w]
        return torch.einsum("nhwmk,mkd->nhwmd", oneHot, k).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, latent):
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
        hard = F.gumbel_softmax((logit / self._scale), 1.0, True)
        quantized = torch.einsum("nhwmk,mkd->nhwmd", logit, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)
        # [n, c, h, w], [n, h, w, m], [n, h, w, m, k]
        return quantized, logit.argmax(-1).byte(), logit, None


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
