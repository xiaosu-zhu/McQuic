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
from mcqc.layers.gdn import GenDivNorm1D
from mcqc.layers.stochastic import gumbelRaoMCK, iGumbelSoftmax


class _resLinear(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        if din > dout:
            nets = [
                nn.Linear(din, dout),
                nn.LeakyReLU(),
                nn.Linear(dout, dout)
            ]
        else:
            nets = [
                nn.Linear(din, din),
                nn.LeakyReLU(),
                nn.Linear(din, dout)
            ]

        self._net = nn.Sequential(*nets)
        if din != dout:
            self._skip = nn.Linear(din, dout)
        else:
            self._skip = None

    def forward(self, x):
        identity = x
        out = self._net(x)
        if self._skip is not None:
            identity = self._skip(x)
        return out + identity


class Mapper(nn.Module):
    def __init__(self, din, k, d):
        super().__init__()
        self._din = din
        self._k = k
        self._d = d
        self._net1 = _resLinear(din, k)
        self._net2 = _resLinear(din, d)

    def forward(self, x):
        # [din, din] -> [din, k]
        x = self._net1(x)
        # [k, din] -> [k, d]
        x = self._net2(x.transpose(0, 1))
        return x


class NonLinearQuantizer(nn.Module):
    def __init__(self, k: int, d: int):
        super().__init__()
        self._k = k
        # dHidden = int(math.sqrt(k * d))
        # self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dHidden, dHidden)))
        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, d)))
        # self._wk = Mapper(dHidden, k, d)
        # self._wv = Mapper(dHidden, k, d)
        # self._wv = _resLinear(d, d)
        # self._wk = _resLinear(d, d)
        self._wv = nn.Linear(d, d, False)
        self._wk = nn.Linear(d, d, False)
        # if doubling:
        #     # self._wvShadow = Mapper(dHidden, k, d)
        #     self._wvShadow = nn.Linear(d, d, False)
        # else:
        #     self._wvShadow = None
        self._temperature1 = nn.Parameter(torch.ones(()))
        # self._temperature2 = nn.Parameter(torch.ones(()))
        self._scale = math.sqrt(d)

    @torch.no_grad()
    def EMAUpdate(self):
        pass

    def encode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        # [k, c]
        k = self._codebook

        # [k, c]
        k = self._wk(k)

        # [n, h, w, k]
        logit = q @ k.permute(1, 0)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1)

    def softEncode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        # [k, c]
        k = self._codebook

        # [k, c]
        k = self._wk(k)

        # [n, h, w, k]
        logit = q @ k.permute(1, 0)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1), logit.softmax(-1)

    def decode(self, code):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        codebook = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = (sample @ codebook).permute(0, 3, 1, 2)
        return quantized

    def softDecode(self, code, soft):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        codebook = self._wv(self._codebook)
        codebookShadow = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = (sample @ codebook).permute(0, 3, 1, 2)
        soft = (soft @ codebookShadow).permute(0, 3, 1, 2)
        return quantized, soft

    def forward(self, latent, temperature):
        q = latent.permute(0, 2, 3, 1)
        k = self._wk(self._codebook)
        # [n, h, w, k]
        logit = (q @ k.permute(1, 0)) / self._scale * self._temperature1
        trueCode = logit.argmax(-1)
        sample = F.gumbel_softmax(logit, temperature, True)
        target = self._wv(self._codebook)
        hard = sample @ target
        hard = hard.permute(0, 3, 1, 2)

        # if self._wvShadow is not None:
        #     softSample = (logit / temperature).softmax(-1)
        #     soft = softSample @ self._wvShadow(self._codebook)
        #     soft = soft.permute(0, 3, 1, 2)
        # else:
        #     soft = hard

        # [n, c, h, w], [n, h, w], [n, h, w, k], [n, c, h, w], [k, c]
        return hard, trueCode, logit


class L2Quantizer(nn.Module):
    def __init__(self, k: int, d: int):
        super().__init__()
        self._k = k
        # dHidden = int(math.sqrt(k * d))
        # self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dHidden, dHidden)))
        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, d)))
        # self._wk = Mapper(dHidden, k, d)
        # self._wv = Mapper(dHidden, k, d)
        # self._wv = _resLinear(d, d)
        # self._wk = _resLinear(d, d)
        # self._wv = nn.Linear(d, d, False)
        # self._wk = nn.Linear(d, d, False)
        # if doubling:
        #     # self._wvShadow = Mapper(dHidden, k, d)
        #     self._wvShadow = nn.Linear(d, d, False)
        # else:
        #     self._wvShadow = None
        self._temperature1 = nn.Parameter(torch.ones(()))
        # self._temperature2 = nn.Parameter(torch.ones(()))
        self._scale = math.sqrt(d)

    @torch.no_grad()
    def EMAUpdate(self):
        pass

    def getLogit(self, x, c):
        # [n, h, w, 1]
        x2 = ((x ** 2).sum(-1))[..., None]
        # [k]
        c2 = (c ** 2).sum(-1)
        # [n, h, w, k]
        inter = x @ c.permute(1, 0)

        distance = -(x2 + c2 - 2 * inter) #.sqrt()

        return distance / self._scale * self._temperature1

    def encode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        # [k, c]
        k = self._codebook

        # [k, c]
        # k = self._wk(k)

        # [n, h, w, k]
        logit = self.getLogit(q, k)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1)

    def softEncode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        # [k, c]
        k = self._codebook

        # [n, h, w, k]
        logit = self.getLogit(q, k)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1), logit.softmax(-1)

    def decode(self, code):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        # codebook = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = (sample @ self._codebook).permute(0, 3, 1, 2)
        return quantized

    def softDecode(self, code, soft):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        # codebook = self._wv(self._codebook)
        # codebookShadow = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = (sample @ self._codebook).permute(0, 3, 1, 2)
        soft = (soft @ self._codebook).permute(0, 3, 1, 2)
        return quantized, soft

    def forward(self, latent, temperature):
        q = latent.permute(0, 2, 3, 1)
        k = self._codebook
        # [n, h, w, k]
        logit = self.getLogit(q, k)
        trueCode = logit.argmax(-1)
        sample = F.gumbel_softmax(logit, temperature, True)
        target = self._codebook
        hard = sample @ target
        hard = hard.permute(0, 3, 1, 2)

        # if self._wvShadow is not None:
        #     softSample = (logit / temperature).softmax(-1)
        #     soft = softSample @ self._wvShadow(self._codebook)
        #     soft = soft.permute(0, 3, 1, 2)
        # else:
        #     soft = hard

        # [n, c, h, w], [n, h, w], [n, h, w, k], [n, c, h, w], [k, c]
        return hard, trueCode, logit


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
        self._temperature1 = nn.Parameter(torch.ones(()))
        # self._temperature2 = nn.Parameter(torch.ones(()))
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
        # n, h, w = frequency.shape
        k, c = value.shape
        soft = (logit / temprature).softmax(-1)
        # needMask = (frequency > (float(h * w) / k)).long()
        # maxFreq, _ = binCount.max(-1, keepdim=True)
        # relaxedFreq = maxFreq + binCount.mean(-1, keepdim=True)
        # # reverse frequencies
        # # max bin -> meanFreq
        # # min bin -> meanFreq + maxbin - minbin
        # # [n, k]
        # reverseBin = relaxedFreq - binCount
        # masked = torch.distributions.Categorical(probs=reverseBin).sample((h, w)).permute(2, 0, 1)
        # sample = trueCode * (1 - needMask) + masked * needMask
        sample = F.one_hot(trueCode, k).float()
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
        logit = (q @ k.permute(1, 0)) / self._scale * self._temperature1
        n, h, w, k = logit.shape
        trueCode = logit.argmax(-1)
        """
        with torch.no_grad():
            if self._dropout:
                # [n, h, w]
                trueCode = logit.argmax(-1)
                # [n, k]
                binCount = torch.zeros((n, k), device=logit.device, dtype=torch.long)
                for i, l in enumerate(trueCode):
                    binCount[i] = torch.bincount(l.flatten(), minlength=k)
                # [n, k] indexed by [n, h, w] -> [n, h, w] frequencies
                ix = torch.arange(n)[:, None, None].expand_as(trueCode)
                frequency = binCount[[ix, trueCode]]
                # [n, h, w]
                dropout = torch.distributions.Bernoulli(frequency / (h * w)).sample().bool()
                # mask = torch.zeros_like(logit, dtype=bool)
                # scatter to mask
                # the max of logit has 0.1 probability to be masked
                mask = torch.zeros_like(logit, dtype=bool).scatter_(-1, trueCode[..., None], dropout[..., None])
            else:
                # [n, h, w]
                trueCode = logit.argmax(-1)
                # [n, k]
                binCount = torch.zeros((n, k), device=logit.device, dtype=torch.long)
                for i, l in enumerate(trueCode):
                    binCount[i] = torch.bincount(l.flatten(), minlength=k)
                # [n, k] indexed by [n, h, w] -> [n, h, w] frequencies
                ix = torch.arange(n)[:, None, None].expand_as(trueCode)
                frequency = binCount[[ix, trueCode]]
                # frequency = torch.ones_like(logit) * logit.shape[0] / logit.numel()
        """
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
        return result, sample, logit, trueCode

    def _argmax(self, q, k, v):
        if self._additionWeight:
            q = self._wq(q)
            k = self._wk(k)
            v = self._wv(v)
        logit = (q @ k.permute(1, 0))
        sample = F.one_hot(logit.argmax(-1), logit.shape[-1]).float()
        result = sample @ v
        trueCode = logit.argmax(-1)
        # frequency = torch.ones_like(logit) * logit.shape[0] / logit.numel()
        return result, sample, logit, trueCode

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
            quantized, sample, logit, trueCode = self._gumbelAttention(latent.permute(0, 2, 3, 1), self._codebook, self._codebook, None, temperature)
        else:
            quantized, sample, logit, trueCode = self._argmax(latent.permute(0, 2, 3, 1), self._codebook, self._codebook)
        quantized = quantized.permute(0, 3, 1, 2)
        # if self._dropout is not None:
        #     quantized = self._dropout(quantized)
        # [n, c, h, w], [n, h, w], [n, h, w, k], [k, c]
        return quantized, trueCode, logit


class Quantizer(nn.Module):
    def __init__(self, m: int, k: int, d: int, dropout: bool = True, deterministic: bool = False, additionWeight: bool = True, ema: float = 0.8):
        super().__init__()
        self._m = m
        d = d // m
        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(m, k, d)))
        if additionWeight:
            self._wq = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(m, d, d)))
            self._wk = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(m, d, d)))
            self._wv = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(m, d, d)))
        self._scale = math.sqrt(d)
        self._additionWeight = additionWeight
        self._deterministic = deterministic
        self._temperature = nn.Parameter(torch.ones(()))
        # if dropout:
        #     self._dropout = PointwiseDropout(0.05)
        # else:
        #     self._dropout = None

    @torch.no_grad()
    def EMAUpdate(self):
        return

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
        code = torch.einsum("nhwmd,mkd->nhwmk", q, k).argmax(-1)
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

    def forward(self, latent, temperature, first=True):
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
            # hard = iGumbelSoftmax((logit / self._scale), temperature, False)
            hard = F.gumbel_softmax(logit / self._scale * self._temperature, temperature, True)
        else:
            # [n, h, w, m, k]
            hard = torch.distributions.OneHotCategorical(logits=logit).sample(())
        quantized = torch.einsum("nhwmk,mkc->nhwmc", hard, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)
        # [n, c, h, w], [n, h, w, m], [n, h, w, m, k]
        return quantized, logit.argmax(-1), logit


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
        code = torch.einsum("nhwmd,mkd->nhwmk", q, k).argmax(-1)
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
        return quantized, logit.argmax(-1), logit
