from contextlib import contextmanager
from typing import List, Tuple, Generator

import torch
from torch import nn
import torch.distributed as dist

# Port from CompressAI
from mcquic.rans import pmfToQuantizedCDF
from mcquic.rans import RansEncoder, RansDecoder

from mcquic.utils.specification import CodeSize


class EntropyCoder(nn.Module):
    def __init__(self, m: int, k: List[int], ema: float = 0.9):
        super().__init__()
        self.encooder = RansEncoder()
        self.decoder = RansDecoder()
        # initial value is uniform
        self._freqEMA = nn.ParameterList(nn.Parameter(torch.ones(m, ki), requires_grad=False) for ki in k) # type: ignore
        self._k = k
        self._decay = 1 - ema

    @torch.no_grad()
    def forward(self, oneHotCodes: List[torch.Tensor]):
        # Update freq by EMA
        # [n, m, h, w, k]
        for lv, code in enumerate(oneHotCodes):
            # [m, k]
            totalCount = code.sum((0, 2, 3))
            # sum over all gpus
            dist.all_reduce(totalCount)

            # up trigger don't perform EMA
            rmsMask = totalCount > self._freqEMA[lv]
            # otherwise
            ema = self._decay * totalCount + (1 - self._decay) * self._freqEMA[lv]

            # update
            self._freqEMA[lv].copy_(totalCount * rmsMask + ema * ~rmsMask)

    @contextmanager
    def readyForCoding(self) -> Generator[List[List[List[int]]], None, None]:
        cdfs = list()
        for freq in self.Freq:
            cdfAtLv = list()
            for freqAtM in freq:
                total = freqAtM.sum()
                if total < 1:
                    prob = torch.ones_like(freqAtM, dtype=torch.float) / len(freqAtM)
                else:
                    prob = freqAtM.float() / total
                cdf = pmfToQuantizedCDF(prob.tolist(), 16)
                cdfAtLv.append(cdf)
            cdfs.append(cdfAtLv)
        try:
            yield cdfs
        finally:
            del cdfs

    @property
    def Freq(self):
        """Yields list of `[m, k]` tensors.

        Yields:
            torch.Tensor: `[m, k]` tensor of level `l`.
        """
        for freqEMA in self._freqEMA:
            # normalize with precision 16bits.
            yield (freqEMA / freqEMA.sum(-1, keepdim=True) * 65536).round().long()

    def _checkShape(self, codes: List[torch.Tensor]):
        info = "Please give codes with correct shape, for example, [[1, 2, 24, 24], [1, 2, 12, 12], ...], which is a `level` length list. each code has shape [n, m, h, w]. "
        if len(codes) < 1:
            raise RuntimeError("Length of codes is 0.")
        n = codes[0].shape[0]
        m = codes[0].shape[1]
        for code in codes:
            newN, newM = code.shape[0], code.shape[1]
            if n < 1:
                raise RuntimeError(info + "Now `n` = 0.")
            if m != newM:
                raise RuntimeError(info + "Now `m` is inconsisitent.")
            if n != newN:
                raise RuntimeError(info + "Now `n` is inconsisitent.")
        return n, m

    @torch.inference_mode()
    def compress(self, codes: List[torch.Tensor], cdfs: List[List[List[int]]]) -> Tuple[List[List[bytes]], List[CodeSize]]:
        """Compress codes to binary.

        Args:
            codes (List[torch.Tensor]): List of tensor, len = level, code.shape = [n, m, h, w]
            cdfs (List[List[List[int]]]): cdfs for entropy coder, len = level, len(cdfs[0]) = m

        Returns:
            List[List[bytes]]: List of binary, len = n, len(binary[0]) = level
            List[CodeSize]]: List of code size, len = n
        """
        n, m = self._checkShape(codes)
        compressed = list(list() for _ in range(n))
        heights = list()
        widths = list()
        # [n, m, h, w]
        for lv, (code, ki, cdf) in enumerate(zip(codes, self._k, cdfs)):
            _, _, h, w = code.shape
            heights.append(h)
            widths.append(w)
            for i, codePerImage in enumerate(code):
                indices = torch.arange(m)[:, None, None]
                # [m, h, w]
                idx = indices.expand_as(codePerImage).flatten().int().tolist()
                cdfSizes = [ki + 2] * m
                # [m, h, w]
                offsets = torch.zeros_like(codePerImage).flatten().int().tolist()
                binary: bytes = self.encooder.encode_with_indexes(codePerImage.flatten().int().tolist(), idx, cdf, cdfSizes, offsets)
                # restored: List[int] = self.decoder.decode_with_indexes(binary, idx, cdf, cdfSizes, offsets)
                # if torch.any(code != torch.tensor(restored, device=code.device).reshape(-1, m, h, w)):
                #     raise RuntimeError("Error")
                # else:
                #     print("Check")
                compressed[i].append(binary)
        return compressed, [CodeSize(m, heights, widths, self._k) for _ in range(n)]

    @torch.inference_mode()
    def decompress(self, binaries: List[List[bytes]], codeSizes: List[CodeSize], cdfs: List[List[List[int]]]) -> List[torch.Tensor]:
        """Restore codes from binary

        Args:
            binaries (List[List[bytes]]): len = n, len(binary[0]) = level
            codeSizes (List[CodeSize]): len = n
            cdfs (List[List[List[int]]]): len = level, len(cdfs[0]) = m

        Returns:
            List[List[torch.Tensor]]: len = level, each code.shape = [n, m, h, w]
        """
        lv = len(binaries[0])
        m = codeSizes[0].m
        codes = list(list() for _ in range(lv))
        indices = torch.arange(m)[:, None, None]
        for i, (binary, codeSize) in enumerate(zip(binaries, codeSizes)):
            codePerImage = list()
            for lv, (binaryAtLv, cdf, ki, h, w) in enumerate(zip(binary, cdfs, self._k, codeSize.heights, codeSize.widths)):
                idx = indices.expand(codeSize.m, h, w).flatten().int().tolist()
                cdfSizes = [ki + 2] * codeSize.m
                offsets = torch.zeros(codeSize.m, h, w, dtype=torch.int).flatten().int().tolist()
                restored: List[int] = self.decoder.decode_with_indexes(binaryAtLv, idx, cdf, cdfSizes, offsets)
                # [m, h, w]
                code = torch.tensor(restored).reshape(codeSize.m, h, w)
                codes[lv].append(code)
        return [torch.stack(c, 0).to(self._freqEMA[0].device) for c in codes]
