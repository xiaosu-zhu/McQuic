import math
from typing import Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from mcquic.modules.entropyCoder import EntropyCoder
from mcquic.nn.base import LowerBound
from mcquic.utils.specification import CodeSize
from mcquic import Consts


class BaseQuantizer(nn.Module):
    _dummyTensor: torch.Tensor
    def __init__(self, m: int, k: List[int]):
        super().__init__()
        self._entropyCoder = EntropyCoder(m, k)
        self._m = m
        self._k = k

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError


    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def readyForCoding(self):
        return self._entropyCoder.readyForCoding()

    def reAssignCodebook(self) -> torch.Tensor:
        raise NotImplementedError

    def syncCodebook(self):
        raise NotImplementedError

    @property
    def Freq(self):
        return self._entropyCoder.Freq

    def compress(self, x: torch.Tensor, cdfs: List[List[List[int]]]) -> Tuple[List[torch.Tensor], List[List[bytes]], List[CodeSize]]:
        codes = self.encode(x)

        # List of binary, len = n, len(binaries[0]) = level
        binaries, codeSize = self._entropyCoder.compress(codes, cdfs)
        return codes, binaries, codeSize

    def _validateCode(self, refCodes: List[torch.Tensor], decompressed: List[torch.Tensor]):
        for code, restored in zip(refCodes, decompressed):
            if torch.any(code != restored):
                raise RuntimeError("Got wrong decompressed result from entropy coder.")

    def decompress(self, binaries: List[List[bytes]], codeSize: List[CodeSize], cdfs: List[List[List[int]]]) -> torch.Tensor:
        decompressed = self._entropyCoder.decompress(binaries, codeSize, cdfs)
        # self._validateCode(codes, decompressed)
        return self.decode(decompressed)


# NOTE: You may notice the quantizer implemented here is different with README.md
#       After some tests, I find some strange behavior if `k` is not placed in the last dim.
#       Generally, although code is neat and output is same as here,
#         training with README's implementation will cause loss become suddenly NAN after a few epoches.
class _multiCodebookQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter, permutationRate: float = 0.15): # type: ignore
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook
        self._scale = math.sqrt(self._k)
        self._temperature = nn.Parameter(torch.ones((self._m, 1, 1, 1))) # type: ignore
        self._bound = LowerBound(Consts.Eps)
        self._permutationRate = permutationRate

    def reAssignCodebook(self, freq: torch.Tensor)-> torch.Tensor:
        codebook = self._codebook.clone().detach()
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
            codebook.data[m, freqGroup < 1] = fullAssigned[selectedIdx]
        # [m, k] bool
        diff = ((codebook - self._codebook) ** 2).sum(-1) > 1e-6
        proportion = diff.flatten()
        self._codebook.data.copy_(codebook)
        return proportion

    def syncCodebook(self):
        # codebook = self._codebook.clone().detach()
        dist.broadcast(self._codebook, 0)

    def encode(self, x: torch.Tensor):
        # [n, m, h, w, k]
        distance = self._distance(x)
        # [n, m, h, w, k] -> [n, m, h, w]
        code = distance.argmin(-1)
        #      [n, m, h, w]
        return code

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def _distance(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        # [n, m, d, h, w]
        x = x.reshape(n, self._m, self._d, h, w)

        # [n, m, 1, h, w]
        x2 = (x ** 2).sum(2, keepdim=True)

        # [m, k, 1, 1]
        c2 = (self._codebook ** 2).sum(-1, keepdim=True)[..., None]
        # [n, m, d, h, w] * [m, k, d] -sum-> [n, m, k, h, w]
        inter = torch.einsum("nmdhw,mkd->nmkhw", x, self._codebook)
        # [n, m, k, h, w]
        distance = x2 + c2 - 2 * inter
        # IMPORTANT to move k to last dim --- PLEASE SEE NOTE.
        # [n, m, h, w, k]
        return distance.permute(0, 1, 3, 4, 2)

    def _logit(self, x: torch.Tensor) -> torch.Tensor:
        logit = -1 * self._distance(x)
        return logit / self._scale

    def _sample(self, x: torch.Tensor, temperature: float):
        # [n, m, h, w, k] * [m, 1, 1, 1]
        logit = self._logit(x) * self._bound(self._temperature)

        # It causes training unstable
        # leave to future tests.
        # add random mask to pick a different index.
        # [n, m, h, w]
        # needPerm = torch.rand_like(logit[..., 0]) < self._permutationRate * rateScale
        # target will set to zero (one of k) but don't break gradient
        # mask = F.one_hot(torch.randint(self._k, (needPerm.sum(), ), device=logit.device), num_classes=self._k).float() * logit[needPerm]
        # logit[needPerm] -= mask.detach()

        # NOTE: STE: code usage is very low; RelaxedOneHotCat: Doesn't have STE trick
        # So reverse back to F.gumbel_softmax
        # posterior = OneHotCategoricalStraightThrough(logits=logit / temperature)
        # [n, m, k, h, w]
        # sampled = posterior.rsample(())

        sampled = F.gumbel_softmax(logit, temperature, True)

        # It causes training unstable
        # leave to future tests.
        # sampled = gumbelArgmaxRandomPerturb(logit, self._permutationRate * rateScale, temperature)
        return sampled, logit

    def forward(self, x: torch.Tensor):
        sample, logit = self._sample(x, 1.0)
        # [n, m, h, w, 1]
        code = logit.argmax(-1, keepdim=True)
        # [n, m, h, w, k]
        oneHot = torch.zeros_like(logit).scatter_(-1, code, 1)
        # [n, m, h, w, k]
        return sample, code[..., 0], oneHot, logit


class _multiCodebookDeQuantization(nn.Module):
    def __init__(self, codebook: nn.Parameter): # type: ignore
        super().__init__()
        self._m, self._k, self._d = codebook.shape
        self._codebook = codebook

    def decode(self, code: torch.Tensor):
        # codes: [n, m, h, w]
        n, _, h, w = code.shape
        # [n, h, w, m]
        code = code.permute(0, 2, 3, 1)
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        ix = torch.arange(self._m, device=code.device).expand_as(code)
        # [n, h, w, m, d]
        indexed = self._codebook[ix, code]
        # [n, c, h, w]
        return indexed.reshape(n, h, w, -1).permute(0, 3, 1, 2)

    # NOTE: ALREADY CHECKED CONSISTENCY WITH NAIVE IMPL.
    def forward(self, sample: torch.Tensor):
        n, m, h, w, k = sample.shape
        # [n, m, h, w, k, 1], [m, 1, 1, k, d] -sum-> [n, m, h, w, d] -> [n, m, d, h, w] -> [n, c, h, w]
        return torch.einsum("nmhwk,mkd->nmhwd", sample, self._codebook).permute(0, 1, 4, 2, 3).reshape(n, -1, h, w)


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

    def syncCodebook(self):
        self._quantizer.syncCodebook()

    def reAssignCodebook(self, freq: torch.Tensor) -> torch.Tensor:
        return self._quantizer.reAssignCodebook(freq)

    def encode(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        code = self._quantizer.encode(self._quantizationHead(z))
        if self._latentHead is None:
            return None, code
        z = self._latentHead(z)
        #      ↓ residual,                         [n, m, h, w]
        return z - self._dequantizer.decode(code), code

    def forward(self, x: torch.Tensor):
        # [h, w] -> [h/2, w/2]
        z = self._latentStageEncoder(x)
        q, code, oneHot, logit = self._quantizer(self._quantizationHead(z))
        if self._latentHead is None:
            return q, None, code, oneHot, logit
        z = self._latentHead(z)
        #         ↓ residual
        return q, z - self._dequantizer(q), code, oneHot, logit

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

    #                [n, m, h, w]
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
            codebook = nn.Parameter(nn.init.normal_(torch.empty(m, ki, channel // m), std=math.sqrt(2 / (5 * channel / m)))) # type: ignore
            quantizer = _multiCodebookQuantization(codebook)
            dequantizer = _multiCodebookDeQuantization(codebook)
            encoders.append(_quantizerEncoder(quantizer, dequantizer, latentStageEncoder, quantizationHead, latentHead))
            decoders.append(_quantizerDecoder(dequantizer, dequantizationHead, sideHead, restoreHead))

        self._encoders: Iterable[_quantizerEncoder] = nn.ModuleList(encoders) # type: ignore
        self._decoders: Iterable[_quantizerDecoder] = nn.ModuleList(decoders) # type: ignore

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        codes = list()
        for encoder in self._encoders:
            x, code = encoder.encode(x) # type: ignore
            #            [n, m, h, w]
            codes.append(code)
        # lv * [n, m, h, w]
        return codes

    def decode(self, codes: List[torch.Tensor]) -> Union[torch.Tensor, None]:
        formerLevel = None
        for decoder, code in zip(self._decoders[::-1], codes[::-1]): # type: ignore
            formerLevel = decoder.decode(code, formerLevel)
        return formerLevel

    def reAssignCodebook(self) -> torch.Tensor:
        freqs = self.Freq
        reassigned: List[torch.Tensor] = list()
        for encoder, freq in zip(self._encoders, freqs):
            # freq: [m, ki]
            reassigned.append(encoder.reAssignCodebook(freq))
        return torch.cat(reassigned).float().mean()

    def syncCodebook(self):
        dist.barrier()
        for encoder in self._encoders:
            encoder.syncCodebook()

    def forward(self, x: torch.Tensor):
        quantizeds = list()
        codes = list()
        oneHots = list()
        logits = list()
        for encoder in self._encoders:
            #          ↓ residual
            quantized, x, code, oneHot, logit = encoder(x)
            # [n, c, h, w]
            quantizeds.append(quantized)
            # [n, m, h, w]
            codes.append(code)
            # [n, m, h, w, k]
            oneHots.append(oneHot)
            # [n, m, h, w, k]
            logits.append(logit)
        formerLevel = None
        for decoder, quantized in zip(self._decoders[::-1], quantizeds[::-1]): # type: ignore
            # ↓ restored
            formerLevel = decoder(quantized, formerLevel)

        # update freq in entropy coder
        self._entropyCoder(oneHots)

        return formerLevel, codes, logits
