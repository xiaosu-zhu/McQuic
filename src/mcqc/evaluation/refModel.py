from typing import OrderedDict, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from mcqc.models.encoder import ResidualAttEncoder, ResidualEncoder
from mcqc.models.decoder import ResidualAttDecoder, ResidualDecoder


class QuantizerEncoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        d = d // m
        self._codebook = nn.Parameter(torch.empty(m, k, d))
        self._wq = nn.Parameter(torch.empty(m, d, d))
        self._wk = nn.Parameter(torch.empty(m, d, d))

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

    @torch.jit.unused
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        codebooks = [c for k, c in state_dict.items() if "_codebook" in k]
        wqs = [c for k, c in state_dict.items() if "_wq" in k]
        wks = [c for k, c in state_dict.items() if "_wk" in k]
        if len(codebooks) != self._codebook.shape[0]:
            raise ValueError(f"Codebook shape mismatch. m in dict: {len(codebooks)}, actually: {self._codebook.shape[0]}")
        for i, c in enumerate(codebooks):
            self._codebook[i] = c
        for i, wq in enumerate(wqs):
            self._wq[i] = wq
        for i, wk in enumerate(wks):
            self._wk[i] = wk

    def forward(self, latent):
        # [n, h, w, m]
        return self.encode(latent)


class QuantizerDecoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        d = d // m
        self._codebook = nn.Parameter(torch.empty(m, k, d))
        self._wv = nn.Parameter(torch.empty(m, d, d))

    def decode(self, codes):
        n, h, w, m = codes.shape
        k = self._codebook.shape[1]
        # [n, h, w, m, k]
        oneHot = F.one_hot(codes.long(), k).float()
        # [m, k, d], [m, d, d] -> [m, k, d]
        v = torch.einsum("mkd,mcd->mkc", self._codebook, self._wv)
        # [n, c, h, w]
        return torch.einsum("nhwmk,mkc->nhwmc", oneHot, v).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    @torch.jit.unused
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        codebooks = [c for k, c in state_dict.items() if "_codebook" in k]
        wvs = [c for k, c in state_dict.items() if "_wv" in k]
        if len(codebooks) != self._codebook.shape[0]:
            raise ValueError(f"Codebook shape mismatch. m in dict: {len(codebooks)}, actually: {self._codebook.shape[0]}")
        for i, c in enumerate(codebooks):
            self._codebook[i] = c
        for i, wv in enumerate(wvs):
            self._wv[i] = wv

    def forward(self, codes):
        # [n, c, h, w]
        return self.decode(codes)


class RefEncoder(nn.Module):
    def __init__(self, m, k, channel):
        super().__init__()
        self._encoder = ResidualAttEncoder(channel, groups=1, alias=False)
        self._quantizer = QuantizerEncoder(m, k, channel)

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        encoderDict = {k[len("_encoder."):]: v for k, v in state_dict.items() if k.startswith("_encoder.")}
        quantizerDict = {k[len("_quantizer."):]: v for k, v in state_dict.items() if k.startswith("_quantizer.")}
        self._encoder.load_state_dict(encoderDict)
        self._quantizer.load_state_dict(quantizerDict)

    @property
    @torch.jit.unused
    def keys(self):
        return "_encoder", "_quantizer"

    def forward(self, x: torch.Tensor) -> Tuple[torch.ByteTensor, torch.IntTensor]:
        x = (x - 0.5) / 0.5
        shape = x.shape
        n, c, h, w = shape
        if c == 1:
            x = x.expand(1, 3, 1, 1)
        hPad = max(0, 32 - h)
        wPad = max(0, 32 - w)
        x = F.pad(x, (0, wPad, 0, hPad))
        return self._quantizer(self._encoder(x)), torch.tensor([h, w], dtype=torch.int)


class RefDecoder(nn.Module):
    def __init__(self, m, k, channel):
        super().__init__()
        self._decoder = ResidualAttDecoder(channel, groups=1)
        self._quantizer = QuantizerDecoder(m, k, channel)

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        encoderDict = {k[len("_decoder."):]: v for k, v in state_dict.items() if k.startswith("_decoder.")}
        quantizerDict = {k[len("_quantizer."):]: v for k, v in state_dict.items() if k.startswith("_quantizer.")}
        self._decoder.load_state_dict(encoderDict)
        self._quantizer.load_state_dict(quantizerDict)

    @property
    @torch.jit.unused
    def keys(self):
        return "_decoder", "_quantizer"

    def forward(self, codes: torch.ByteTensor, shape: torch.IntTensor) -> torch.Tensor:
        h, w = shape[0], shape[1]
        return ((self._decoder(self._quantizer(codes)))[..., :h, :w].tanh() + 1) / 2
