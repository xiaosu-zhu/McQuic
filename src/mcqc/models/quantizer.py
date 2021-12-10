import math

import torch
from torch import nn
import torch.nn.functional as F


class L2Quantizer(nn.Module):
    def __init__(self, k: int, dIn: int, dHidden: int):
        super().__init__()
        self._k = k
        # dHidden = int(math.sqrt(k * d))
        # self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(dHidden, dHidden)))
        self._codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(k, dHidden)))
        self._wv = nn.Linear(dIn, dHidden)
        self._wq = nn.Linear(dHidden, dIn)
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
        self._scale = math.sqrt(k)
        self._eps = 1e-7

    @torch.no_grad()
    def EMAUpdate(self):
        pass

    def getLogit(self, x, c):
        # [n, h, w, 1]
        x2 = (x ** 2).sum(-1, keepdim=True)
        # [k]
        c2 = (c ** 2).sum(-1)
        # [n, h, w, k]
        inter = x @ c.permute(1, 0)

        distance = -(x2 + c2 - 2 * inter) #.sqrt()

        return distance

    def encode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        q = self._wv(q)


        # [k, c]
        k = self._codebook

        # [k, c]
        # k = self._wk(k)

        # [n, h, w, k]
        logit = self.getLogit(q, k)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1)

    def rawAndQuantized(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        q = self._wv(q)

        # [k, c]
        k = self._codebook

        # [k, c]
        # k = self._wk(k)

        # [n, h, w, k]
        logit = self.getLogit(q, k)

        sample = F.one_hot(logit.argmax(-1), self._k).float()

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1), latent, (sample @ self._codebook).permute(0, 3, 1, 2)

    def softEncode(self, latent):
        # [n, h, w, c]
        q = latent.permute(0, 2, 3, 1)
        q = self._wv(q)

        # [k, c]
        k = self._codebook

        # [n, h, w, k]
        logit = self.getLogit(q, k)

        # sample = F.gumbel_softmax(logit, 1.0, True)
        return logit.argmax(-1), logit.softmax(-1)

    def decode(self, code):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        quantized = (sample @ self._codebook)
        # codebook = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        result = self._wq(quantized).permute(0, 3, 1, 2)
        return result, quantized.permute(0, 3, 1, 2)

    def softDecode(self, code, soft):
        # [n, h, w, k]
        sample = F.one_hot(code, self._k).float()
        # codebook = self._wv(self._codebook)
        # codebookShadow = self._wv(self._codebook)
        # [n, h, w, c] -> [n, c, h, w]
        quantized = self._wq((sample @ self._codebook)).permute(0, 3, 1, 2)
        soft = self._wq((soft @ self._codebook)).permute(0, 3, 1, 2)
        return quantized, soft

    def forward(self, latent, temperature):
        q = latent.permute(0, 2, 3, 1)
        raw = self._wv(q)
        k = self._codebook

        # [n, h, w, k]
        logitRaw = self.getLogit(raw, k)
        logit = logitRaw / (self._temperature1 * temperature)
        trueCode = logit.argmax(-1)
        sample = F.gumbel_softmax(logit, math.sqrt(temperature), True)
        code = sample.argmax(-1)
        target = self._codebook
        quantized = sample @ target

        hard = self._wq(quantized)
        hard = hard.permute(0, 3, 1, 2)

        # if self._wvShadow is not None:
        #     softSample = (logit / temperature).softmax(-1)
        #     soft = softSample @ self._wvShadow(self._codebook)
        #     soft = soft.permute(0, 3, 1, 2)
        # else:
        #     soft = hard

        # [n, c, h, w], [n, h, w], [n, h, w, k], [n, h, w, c], [k, c]
        return hard, code, trueCode, logitRaw, (raw, quantized), self._codebook


class QuantizerEncoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        # self._

    def forward(self, x):
        pass
