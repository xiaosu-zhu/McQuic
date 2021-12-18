import math
from typing import Callable, Dict, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import OneHotCategoricalStraightThrough

from mcqc.layers.convs import conv1x1


class CascadedQuantization(nn.Module):
    def __init__(self, channel: int, level: int, groups: int, k: int):
        super().__init__()


class VarianceForward(nn.Module):
    def __init__(self):
        super().__init__()
        self._originalMapping = None
        self._quantizedMapping = None

    def forward(self, original, quantized):
        pass


class VarianceBackward(nn.Module):
    def __init__(self):
        super().__init__()


class _multiCodebookQuantization(nn.Module):
    def __init__(self, channel: int, m: int, k: int):
        super().__init__()
        self._m = m
        self._k = k
        self._logistic = conv1x1(channel, m * k, bias=False, groups=m)

    def forward(self, x: torch.Tensor):
        # [n, m * k, h, w]
        logit = self._logistic(x)
        n, _, h, w = logit.shape
        # [n, m, h, w, k]
        posterior = OneHotCategoricalStraightThrough(logits=logit.reshape(n, self._m, self._k, h, w).permute(0, 1, 3, 4, 2))
        # [n, m, h, w, k]
        quantized = posterior.rsample(())
        # [n, m, h, w]
        code = quantized.argmax(-1)
        #      [n, m * k, h, w]
        return quantized.permute(0, 1, 4, 2, 3).reshape(n, -1, h, w), code, logit
class _multiCodebookDeQuantization(nn.Module):
    def __init__(self, channel: int, m: int, k: int):
        super().__init__()
        self._m = m
        self._k = k
        self._mapping = conv1x1(m * k, channel, bias=False, groups=m)

    def forward(self, x: torch.Tensor):
        # [n, k, h, w]
        return self._mapping(x)


class _quantizerEncoder(nn.Module):
    """
    Default structure:
    ```plain
        x [H, W]
        | `latentStageEncoder`
        z [H/2 , W/2] -------╮
        | `quantizationHead` | `latentHead`
        q [H/2, W/2]         z [H/2, w/2]
        |                    |
        ├-`subtract` --------╯
        residual for next level
    ```
    """

    def __init__(self, quantizer: nn.Module, dequantizer: nn.Module, latentStageEncoder: nn.Module, quantizationHead: nn.Module, latentHead: Union[None, nn.Module]):
        super().__init__()
        self._quantizer =  quantizer
        self._dequantizer =  dequantizer
        self._latentStageEncoder =  latentStageEncoder
        self._quantizationHead =  quantizationHead
        self._latentHead =  latentHead

    def forward(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        q, code, logit = self._quantizer(self._quantizationHead(z))
        if self._latentHead is None:
            return q, None, code, logit
        z = self._latentHead(z)
        #         ↓ residual
        return q, z - self._dequantizer(q), code, logit

class _quantizerDecoder(nn.Module):
    """
    Default structure:
    ```plain
        q [H/2, W/2]            formerLevelRestored [H/2, W/2]
        | `dequantizaitonHead`  | `sideHead`
        ├-`add` ----------------╯
        xHat [H/2, W/2]
        | `restoreHead`
        nextLevelRestored [H, W]
    ```
    """

    def __init__(self, dequantizer: nn.Module, dequantizationHead: nn.Module, sideHead: Union[None, nn.Module], restoreHead: nn.Module):
        super().__init__()
        self._dequantizer =  dequantizer
        self._dequantizationHead =  dequantizationHead
        self._sideHead =  sideHead
        self._restoreHead =  restoreHead

    def forward(self, q: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer(q))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)

class UMGMQuantizer(nn.Module):
    _components = [
        "latentStageEncoder",
        "quantizationHead",
        "latentHead",
        "dequantizationHead",
        "sideHead",
        "restoreHead"
    ]
    def __init__(self, channel: int, m: int, k: Union[int, List[int]], components: Dict[str, Callable[[], nn.Module]]):
        super().__init__()
        if isinstance(k, int):
            k = [k]
        componentFns = [components[key] for key in self._components]
        latentStageEncoderFn, quantizationHeadFn, latentHeadFn, dequantizationHeadFn, sideHeadFn, restoreHeadFn = componentFns

        encoders = list()
        decoders = list()

        for i, ki in enumerate(k):
            latentStageEncoder = latentStageEncoderFn()
            quantizationHead = quantizationHeadFn()
            latentHead = latentHeadFn() if i < len(k) - 1 else None
            dequantizationHead = dequantizationHeadFn()
            sideHead = sideHeadFn() if i > 0 else None
            restoreHead = restoreHeadFn()
            quantizer = _multiCodebookQuantization(channel, m, ki)
            dequantizer = _multiCodebookDeQuantization(channel, m, ki)
            encoders.append(_quantizerEncoder(quantizer, dequantizer, latentStageEncoder, quantizationHead, latentHead))
            decoders.append(_quantizerDecoder(dequantizer, dequantizationHead, sideHead, restoreHead))

        self._encoders = nn.ModuleList(encoders)
        self._decoders = nn.ModuleList(decoders)

    def forward(self, x: torch.Tensor):
        quantizeds = list()
        codes = list()
        logits = list()
        for encoder in self._encoders:
            #          ↓ residual
            quantized, x, code, logit = encoder(x)
            quantizeds.append(quantized)
            codes.append(code)
            logits.append(logit)
        formerLevel = None
        for decoder, quantized in zip(self._decoders, quantizeds[::-1]):
            # ↓ restored
            formerLevel = decoder(quantized, formerLevel)
        return formerLevel, codes, logits

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
