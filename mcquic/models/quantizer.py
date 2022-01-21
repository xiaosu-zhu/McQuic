import math
from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import OneHotCategoricalStraightThrough

from mcquic.nn import NonNegativeParametrizer, LogExpMinusOne, conv1x1
from mcquic.models.entropyCoder import EntropyCoder
from mcquic.utils.specification import CodeSize


class BaseQuantizer(nn.Module):
    _dummyTensor: torch.Tensor
    def __init__(self, m: int, k: List[int]):
        super().__init__()
        self.register_buffer("_dummyTensor", torch.empty(()))
        self._entropyCoder = EntropyCoder(m, k)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def count(self, x: torch.Tensor):
        self._entropyCoder.updateFreq(self.encode(x))

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def readyForCoding(self):
        return self._entropyCoder.readyForCoding()

    def clearFreq(self):
        return self._entropyCoder.clearFreq()

    def reAssignCodebook(self):
        raise NotImplementedError

    def syncCodebook(self) -> float:
        raise NotImplementedError

    @property
    def Freq(self):
        return self._entropyCoder.Freq

    def compress(self, x: torch.Tensor, cdfs: List[List[List[int]]]) -> Tuple[List[torch.Tensor], List[List[bytes]], List[CodeSize]]:
        codes = self.encode(x)
        binaries, codeSize = self._entropyCoder.compress(codes, cdfs)
        return codes, binaries, codeSize

    def _validateCode(self, codes: List[torch.Tensor], decompressed: List[torch.Tensor]):
        for code, restored in zip(codes, decompressed):
            if torch.any(code != restored):
                raise RuntimeError("Got wrong decompressed result from entropy coder.")

    def decompress(self, codes: List[torch.Tensor], binaries: List[List[bytes]], codeSize: List[CodeSize], cdfs: List[List[List[int]]]) -> torch.Tensor:
        decompressed = self._entropyCoder.decompress(binaries, codeSize, cdfs)
        decompressed = [c.to(self._dummyTensor.device) for c in decompressed]
        self._validateCode(codes, decompressed)
        return self.decode(decompressed)


class _multiCodebookQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._preProcess = conv1x1(self._m * self._d, self._m * self._d, groups=self._m)
        self._wC = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self._m, self._d, self._d)))
        self._scale = math.sqrt(self._k)
        self._codebook = codebook
        # self._logExpMinusOne = LogExpMinusOne()
        self._zeroBound = NonNegativeParametrizer(1e-6)
        self._temperature = nn.Parameter(torch.ones((self._m)))

    def reAssignCodebook(self, freq: torch.Tensor):
        freq = freq.to(self._codebook.device).clone().detach()
        #       [k, d],        [k]
        for m, (codebookGroup, freqGroup) in enumerate(zip(self._codebook, freq)):
            neverAssigned = codebookGroup[freqGroup < 1]
            if len(neverAssigned) > self._k // 2:
                mask = torch.zeros((len(neverAssigned), ), dtype=torch.long, device=self._codebook.device)
                maskIdx = torch.randperm(len(mask))[self._k // 2:]
                mask[maskIdx] = 1
                freqGroup[freqGroup < 1] = mask
                neverAssigned = codebookGroup[freqGroup < 1]
            argIdx = torch.argsort(freqGroup, descending=True)[:(self._k - len(neverAssigned))]
            fullAssigned = codebookGroup[argIdx]
            selectedIdx = torch.randperm(len(fullAssigned))[:len(neverAssigned)]
            self._codebook.data[m, freqGroup < 1] = fullAssigned[selectedIdx]

    def syncCodebook(self) -> float:
        codebook = self._codebook.clone().detach()
        dist.broadcast(codebook, 0)
        diff = codebook != self._codebook
        proportion = diff.float().mean().item()
        self._codebook.data.copy_(codebook)
        return proportion

    def encode(self, x: torch.Tensor):
        # [n, m, h, w, k]
        distance = self._distance(self._preProcess(x))
        # [n, m, h, w, k] -> [n, m, h, w]
        code = distance.argmin(-1)
        #      [n, m, h, w]
        return code

    def _codebookMapped(self) -> torch.Tensor:
        return torch.einsum("mkd,mcd->mkc", self._codebook, self._wC)

    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        # [n, m, d, h, w]
        x = x.reshape(n, self._m, self._d, h, w)

        # [n, m, 1, h, w]
        x2 = (x ** 2).sum(2, keepdim=True)
        # [m, k, 1, 1]
        c2 = (self._codebook ** 2).sum(-1, keepdim=True)[..., None]
        # [n, m, d, h, w] * [m, k, d] -sum-> [n, m, k, h, w]
        inter = torch.einsum("nmdhw,mkd->nmkhw", x, self._codebookMapped())
        # [n, m, k, h, w]
        distance = x2 + c2 - 2 * inter
        # [n, m, h, w, k]
        return distance.sqrt().permute(0, 1, 3, 4, 2)

        # [n, m, d, h, w] -> [m, n, h, w, d]
        x = x.reshape(n, self._m, self._d, h, w).permute(1, 0, 3, 4, 2)

        # [m, n, h, w, 1]
        x2 = (x ** 2).sum(-1, keepdim=True)
        # [m, k]
        c2 = (self._codebook ** 2).sum(-1)
        distances = list()
        for m in range(len(self._codebook)):
            # [n, h, w, d]
            xm = x[m]
            # [k, d]
            cm = self._codebook[m]
            # [n, h, w, k]
            inter = xm @ cm.T
            # [n, h, w, 1] + [k] - 2 * [n, h, w, k]
            distance = x2[m] + c2[m] - 2 * inter
            distances.append(distance)

        # m * [n, h, w, k] -> [n, m, h, w, k]
        distance = torch.stack(distances, 1)
        # [n, m, h, w, k]
        return distance

    def _logit(self, x: torch.Tensor) -> torch.Tensor:
        # ensure > 0
        # distance = self._distanceBound(self._distance(self._preProcess(x)).exp() - 1)
        # map to -∞ ~ +∞
        logit = -1 * self._distance(self._preProcess(x))
        return logit

    def _sample(self, x: torch.Tensor):
        # [n, m, h, w, k] * [m, 1, 1, 1]
        logit = self._logit(x) * self._zeroBound(self._temperature)[:, None, None, None]
        posterior = OneHotCategoricalStraightThrough(logits=logit)
        # [n, m, h, w, k]
        sampled = posterior.rsample(())
        return sampled, logit

    def forward(self, x: torch.Tensor):
        sample, logit = self._sample(x)
        # [n, m, h, w]
        code = sample.argmax(-1)
        #      [n, m, h, w, k]
        return sample, code, logit


class _multiCodebookDeQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter):
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self._postProcess = conv1x1(self._m * self._d, self._m * self._d, groups=self._m)
        self._wC = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self._m, self._d, self._d)))

    def _codebookMapped(self) -> torch.Tensor:
        return torch.einsum("mkd,mcd->mkc", self._codebook, self._wC)

    def decode(self, code: torch.Tensor):
        # codes: [n, m, h, w]
        n, _, h, w = code.shape
        # [n, h, w, m]
        code = code.permute(0, 2, 3, 1)
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        ix = torch.arange(self._m, device=code.device).expand_as(code)
        # [n, h, w, m, d]
        indexed = self._codebookMapped()[ix, code]
        # [n, c, h, w]
        return self._postProcess(indexed.reshape(n, h, w, -1).permute(0, 3, 1, 2))
        # n, m, h, w = code.shape
        # # [n, m, h, w, k]
        # oneHot = F.one_hot(code, self._k)
        # # [n, m, h, w, k, 1], [m, 1, 1, k, d] -sum-> [n, m, h, w, d]
        # return (oneHot[..., None] * self._codebook[:, None, None, ...]).sum(-2)

    def forward(self, sample: torch.Tensor):
        n, m, h, w, k = sample.shape
        # [n, m, h, w, k, 1], [m, 1, 1, k, d] -sum-> [n, m, h, w, d] -> [n, m, d, h, w] -> [n, c, h, w]
        return self._postProcess(torch.einsum("nmhwk,mkd->nmhwd", sample, self._codebookMapped()).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w))

        quantizeds = list()
        for i in range(len(self._codebook)):
            # [n, h, w, k]
            oneHot = sample[:, i]
            # [n, h, w, k] @ [k, d] -> [n, h, w, d]
            quantized = oneHot @ self._codebook[i]
            quantizeds.append(quantized)
        # m * [n, h, w, d] -> [n, h, w, c] -> [n, c, h, w]
        return torch.cat(quantizeds, -1).permute(0, 3, 1, 2)


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

    def __init__(self, quantizer: _multiCodebookQuantization, dequantizer: _multiCodebookDeQuantization, latentStageEncoder: nn.Module, quantizationHead: nn.Module, latentHead: Union[None, nn.Module]):
        super().__init__()
        self._quantizer = quantizer
        self._dequantizer = dequantizer
        self._latentStageEncoder = latentStageEncoder
        self._quantizationHead = quantizationHead
        self._latentHead = latentHead

    def syncCodebook(self) -> float:
        return self._quantizer.syncCodebook()

    def reAssignCodebook(self, freq: torch.Tensor):
        self._quantizer.reAssignCodebook(freq)

    def encode(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        code = self._quantizer.encode(self._quantizationHead(z))
        if self._latentHead is None:
            return None, code
        z = self._latentHead(z)
        #      ↓ residual
        return z - self._dequantizer.decode(code), code

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

    def __init__(self, dequantizer: _multiCodebookDeQuantization, dequantizationHead: nn.Module, sideHead: Union[None, nn.Module], restoreHead: nn.Module):
        super().__init__()
        self._dequantizer =  dequantizer
        self._dequantizationHead =  dequantizationHead
        self._sideHead =  sideHead
        self._restoreHead =  restoreHead

    def decode(self, code: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer.decode(code))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)

    def forward(self, q: torch.Tensor, formerLevel: Union[None, torch.Tensor]):
        q = self._dequantizationHead(self._dequantizer(q))
        if self._sideHead is not None:
            xHat = q + self._sideHead(formerLevel)
        else:
            xHat = q
        return self._restoreHead(xHat)


class UMGMQuantizer(BaseQuantizer):
    _components = [
        "latentStageEncoder",
        "quantizationHead",
        "latentHead",
        "dequantizationHead",
        "sideHead",
        "restoreHead"
    ]
    def __init__(self, channel: int, m: int, k: Union[int, List[int]], components: Dict[str, Callable[[], nn.Module]]):
        if isinstance(k, int):
            k = [k]
        super().__init__(m, k)
        componentFns = [components[key] for key in self._components]
        latentStageEncoderFn, quantizationHeadFn, latentHeadFn, dequantizationHeadFn, sideHeadFn, restoreHeadFn = componentFns

        encoders = list()
        decoders = list()

        for i, ki in enumerate(k):
            latentStageEncoder = latentStageEncoderFn()
            quantizationHead = quantizationHeadFn()
            latentHead = latentHeadFn() if i < len(k) - 1 else None
            dequantizationHead = dequantizationHeadFn()
            sideHead = sideHeadFn() if i < len(k) - 1 else None
            restoreHead = restoreHeadFn()
            codebook = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(m, ki, channel // m)))
            quantizer = _multiCodebookQuantization(codebook)
            dequantizer = _multiCodebookDeQuantization(codebook)
            encoders.append(_quantizerEncoder(quantizer, dequantizer, latentStageEncoder, quantizationHead, latentHead))
            decoders.append(_quantizerDecoder(dequantizer, dequantizationHead, sideHead, restoreHead))

        self._encoders: Iterable[_quantizerEncoder] = nn.ModuleList(encoders) # type: ignore
        self._decoders: Iterable[_quantizerDecoder] = nn.ModuleList(decoders) # type: ignore

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        codes = list()
        for encoder in self._encoders:
            x, code = encoder.encode(x)
            codes.append(code)
        return codes

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        formerLevel = None
        for decoder, code in zip(self._decoders[::-1], codes[::-1]):
            formerLevel = decoder.decode(code, formerLevel)
        return formerLevel

    def reAssignCodebook(self):
        freqs = self.Freq
        for encoder, freq in zip(self._encoders, freqs):
            # freq: [m, ki]
            encoder.reAssignCodebook(freq)

    def syncCodebook(self) -> float:
        dist.barrier()
        proportions: List[float] = list()
        for encoder in self._encoders:
            proportions.append(encoder.syncCodebook())
        return sum(proportions) / len(proportions)

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
        for decoder, quantized in zip(self._decoders[::-1], quantizeds[::-1]):
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
