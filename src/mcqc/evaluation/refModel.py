from typing import List, OrderedDict, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from mcqc.models.encoder import Director, DownSampler, EncoderHead, ResidualAttEncoder, ResidualBaseEncoder, ResidualEncoder
from mcqc.models.decoder import ResidualAttDecoder, ResidualBaseDecoder, ResidualDecoder, UpSampler



class Preprocess(nn.Module):
    def __init__(self, base: int = 128):
        super().__init__()
        self._base = base

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        x = (x - 0.5) / 0.5
        wPad = math.ceil(w / self._base) * self._base - w
        hPad = math.ceil(h / self._base) * self._base - h
        padLeft = wPad // 2
        padRight = wPad - padLeft
        padTop = hPad // 2
        padBottom = hPad - padTop

        x = F.pad(x, (padLeft, padRight, padTop, padBottom), mode="reflect")
        if c == 1:
            n, c, h, w = x.shape
            x = x.expand(n, 3, h, w)
        return x, torch.tensor([padLeft, padRight, padTop, padBottom])


class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, padding):
        pass


class QuantizerEncoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        self._codebook = nn.Parameter(torch.empty(m, k, d))

    def encode(self, latent):
        n, _, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        # [n, h, w, m, 1]
        q2 = (q ** 2).sum(-1, keepdim=True)
        # [m, k]
        c2 = (self._codebook ** 2).sum(-1)

        inter = torch.einsum("nhwmd,mkd->nhwmk", q, self._codebook)

        # [n, h, w, m, k]
        distance = -(q2 + c2 - 2 * inter)
        return distance.argmax(-1)

    @torch.jit.unused
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        codebooks = [c for k, c in state_dict.items() if "_codebook" in k]
        if len(codebooks) != self._codebook.shape[0]:
            raise ValueError(f"Codebook shape mismatch. m in dict: {len(codebooks)}, actually: {self._codebook.shape[0]}")
        for i, c in enumerate(codebooks):
            self._codebook[i] = c

    def forward(self, latent):
        # [n, h, w, m]
        return self.encode(latent)


class QuantizerDecoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        self._codebook = nn.Parameter(torch.empty(m, k, d))
        self.register_buffer("_ix", torch.arange(m))

    def decode(self, codes: torch.Tensor):
        # codes: [n, h, w, m]
        n, h, w, _ = codes.shape
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        ix = self._ix.expand_as(codes)
        return self._codebook[[ix, codes]].reshape(n, h, w, -1).permute(0, 3, 1, 2)


    @torch.jit.unused
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        codebooks = [c for k, c in state_dict.items() if "_codebook" in k]
        if len(codebooks) != self._codebook.shape[0]:
            raise ValueError(f"Codebook shape mismatch. m in dict: {len(codebooks)}, actually: {self._codebook.shape[0]}")
        for i, c in enumerate(codebooks):
            self._codebook[i] = c

    def forward(self, codes):
        # [n, c, h, w]
        return self.decode(codes)


class RefEncoder(nn.Module):
    def __init__(self, m, k, channel, groups, alias):
        super().__init__()
        self._levels = len(k)

        self._encoder = ResidualBaseEncoder(channel, groups, alias)

        self._heads = nn.ModuleList(EncoderHead(channel, 1, alias) for _ in range(self._levels))
        self._mappers = nn.ModuleList(DownSampler(channel, 1, alias) for _ in range(self._levels - 1))
        self._quantizers = nn.ModuleList(QuantizerEncoder(m, k, channel // m) for _ in k)
        self._deQuantizers = nn.ModuleList(QuantizerDecoder(m, k, channel // m) for _ in k)

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        encoderDict = {k[len("_encoder."):]: v for k, v in state_dict.items() if k.startswith("_encoder.")}
        headsDict = {k[len("_heads."):]: v for k, v in state_dict.items() if k.startswith("_heads.")}
        mappersDict = {k[len("_mappers."):]: v for k, v in state_dict.items() if k.startswith("_mappers.")}
        quantizerDict = {k[len("_quantizer."):]: v for k, v in state_dict.items() if k.startswith("_quantizer.")}
        self._encoder.load_state_dict(encoderDict)
        self._heads.load_state_dict(headsDict)
        self._mappers.load_state_dict(mappersDict)
        self._quantizers.load_state_dict(quantizerDict)
        self._deQuantizers.load_state_dict(quantizerDict)

    @property
    @torch.jit.unused
    def keys(self):
        return "_encoder", "_quantizer"

    def forward(self, x: torch.Tensor) -> List[torch.LongTensor]:
        codes = list()
        latent = self._encoder(x)
        for i in range(self._levels):
            head = self._heads[i]
            z = head(latent)
            if i < self._levels - 1:
                mapper = self._mappers[i]
                latent = mapper(latent)
                code = self._quantizers[i](z)
                hard = self._deQuantizers[i](code)
                latent = latent - hard
            else:
                code = self._quantizers[i](z)
            codes.append(code)
        return codes


class RefDecoder(nn.Module):
    def __init__(self, m, k, channel, alias):
        super().__init__()
        self._levels = len(k)

        self._decoder = ResidualBaseDecoder(channel, 1)

        self._reverses = nn.ModuleList(UpSampler(channel, 1, alias) for _ in range(self._levels))
        self._scatters = nn.ModuleList(Director(channel, 1, alias) for _ in range(self._levels - 1))
        self._quantizers = nn.ModuleList(QuantizerDecoder(m, k, channel // m) for _ in k)

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        encoderDict = {k[len("_decoder."):]: v for k, v in state_dict.items() if k.startswith("_decoder.")}
        reversesDict = {k[len("_reverses."):]: v for k, v in state_dict.items() if k.startswith("_reverses.")}
        scattersDict = {k[len("_scatters."):]: v for k, v in state_dict.items() if k.startswith("_scatters.")}
        quantizerDict = {k[len("_quantizer."):]: v for k, v in state_dict.items() if k.startswith("_quantizer.")}
        self._decoder.load_state_dict(encoderDict)
        self._quantizers.load_state_dict(quantizerDict)
        self._reverses.load_state_dict(reversesDict)
        self._scatters.load_state_dict(scattersDict)

    @property
    @torch.jit.unused
    def keys(self):
        return "_decoder", "_quantizer"

    def forward(self, codes: torch.ByteTensor) -> torch.Tensor:
        smallQ = self._reverses[-1](self._quantizers[-1](codes[-1]))
        for i in range(self._levels - 1, -1, -1):
            q = self._scatters[i](self._quantizers[i](codes[i]))
            smallQ = self._reverses[i](q + smallQ)
        return self._decoder(smallQ).tanh()
