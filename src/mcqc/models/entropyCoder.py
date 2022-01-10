from contextlib import contextmanager
from typing import List, Tuple, Generator

import torch
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans
from vlutils.base.restorable import Restorable

from mcqc.utils.specification import CodeSize


class EntropyCoder(Restorable):
    def __init__(self, m: int, k: List[int]):
        self.encooder = ans.RansEncoder()
        self.decoder = ans.RansDecoder()

        super().__init__()
        self._freq = list(torch.zeros(m, ki, dtype=torch.int) for ki in k)
        self._k = k

    def clearFreq(self):
        self._freq = list(torch.zeros_like(x) for x in self._freq)

    @torch.no_grad()
    def updateFreq(self, codes: List[torch.Tensor]):
        # [n, m, h, w]
        for lv, code in enumerate(codes):
            code = code.detach().cpu()
            # [n, h, w]
            for m, codeAtM in enumerate(code.permute(1, 0, 2, 3)):
                self._freq[lv][m] += torch.bincount(codeAtM.flatten(), minlength=len(self._freq[lv][m]))

    @contextmanager
    def readyForCoding(self) -> Generator[List[List[List[int]]], None, None]:
        cdfs = list()
        for freq in self._freq:
            cdfAtLv = list()
            for freqAtM in freq:
                total = freqAtM.sum()
                if total < 1:
                    prob = torch.ones_like(freqAtM, dtype=torch.float) / len(freqAtM)
                else:
                    prob = freqAtM.float() / total
                cdf = pmf_to_quantized_cdf(prob.tolist(), 16)
                cdfAtLv.append(cdf)
            cdfs.append(cdfAtLv)
        try:
            yield cdfs
        finally:
            del cdfs

    def _checkShape(self, codes: List[torch.Tensor]):
        if len(codes) < 1:
            raise RuntimeError("Length of codes is 0.")
        m = codes[0].shape[1]
        for code in codes:
            n, newM = code.shape[0], code.shape[1]
            if n < 1:
                raise RuntimeError("Please give codes with correct shape, for example, [[1, 2, 24, 24], [1, 2, 12, 12], ...]. Now `batch` = 0.")
            if m != newM:
                raise RuntimeError("Please give codes with correct shape, for example, [[1, 2, 24, 24], [1, 2, 12, 12], ...]. Now `m` is not a constant.")

    @torch.inference_mode()
    def compress(self, codes: List[torch.Tensor], cdfs: List[List[List[int]]]) -> Tuple[List[bytes], CodeSize]:
        compressed = list()
        self._checkShape(codes)
        heights = list()
        widths = list()
        # [n, m, h, w]
        for lv, (code, ki, cdf) in enumerate(zip(codes, self._k, cdfs)):
            n, m, h, w = code.shape
            indices = torch.arange(m)[:, None, None]
            heights.append(h)
            widths.append(w)
            idx = indices.expand_as(code).flatten().int().tolist()
            cdfSizes = [ki + 2] * m
            offsets = torch.zeros_like(code).flatten().int().tolist()
            binary: bytes = self.encooder.encode_with_indexes(code.flatten().int().tolist(), idx, cdf, cdfSizes, offsets)
            compressed.append(binary)
        return compressed, CodeSize(m, heights, widths, self._k)

    @torch.inference_mode()
    def decompress(self, binaries: List[bytes], codeSize: CodeSize, cdfs: List[List[List[int]]]) -> List[torch.Tensor]:
        codes = list()
        indices = torch.arange(codeSize.m)[:, None, None]
        for lv, (binary, h, w, ki, cdf) in enumerate(zip(binaries, codeSize.heights, codeSize.widths, self._k, cdfs)):
            idx = indices.expand(codeSize.m, h, w).flatten().int().tolist()
            # [k + 2] * m
            cdfSizes = [ki + 2] * codeSize.m
            offsets = torch.zeros(codeSize.m, h, w, dtype=torch.int).flatten().int().tolist()
            restored: List[int] = self.decoder.decode_with_indexes(binary, idx, cdf, cdfSizes, offsets)
            # [1, m, h, w]
            code = torch.tensor(restored).reshape(-1, codeSize.m, h, w)
            codes.append(code)
        return codes
