from typing import List, OrderedDict, Tuple
import math

import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F

from mcquic.models.deprecated.encoder import BaseEncoder5x5, Director, Director5x5, DownSampler, DownSampler5x5, EncoderHead, EncoderHead5x5, ResidualBaseEncoder
from mcquic.models.deprecated.decoder import BaseDecoder5x5, ResidualBaseDecoder, UpSampler, UpSampler5x5



class Preprocess(nn.Module):
    def __init__(self, base: int = 128):
        super().__init__()
        self._base = base

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if padRight == 0:
            padRight = -w
        if padBottom == 0:
            padBottom = -h
        return x, torch.tensor([c, padLeft, padRight, padTop, padBottom])


class PostProcess(nn.Module):
    def forward(self, x: torch.Tensor, cAndPadding: torch.Tensor) -> torch.Tensor:
        x = x[:, :, cAndPadding[3]:(-cAndPadding[4]), cAndPadding[1]:(-cAndPadding[2])]
        if cAndPadding[0] == 1:
            x = x.mean(1, keepdim=True)
        return (x + 1) / 2


class QuantizerEncoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        """[summary]

        Args:
            m (int): [description]
            k (int): [description]
            d (int): [description]
        """
        super().__init__()
        self._m = m
        self._wv = nn.Parameter(torch.empty(m, d, d))
        self._bv = nn.Parameter(torch.empty(m, d))
        # self._wq = nn.Parameter(torch.empty(m, d, d))
        # self._bq = nn.Parameter(torch.empty(m, d))
        self._codebook = nn.Parameter(torch.empty(m, k, d))

    def encode(self, latent):
        n, _, h, w = latent.shape
        # [n, h, w, m, d]
        q = latent.permute(0, 2, 3, 1).reshape(n, h, w, self._m, -1)
        q = torch.einsum("nhwmd,mcd->nhwmc", q, self._wv) + self._bv
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
        wvs = [c for k, c in state_dict.items() if "_wv.weight" in k]
        bvs = [c for k, c in state_dict.items() if "_wv.bias" in k]
        for i, (w, b) in enumerate(zip(wvs, bvs)):
            self._wv[i] = w
            self._bv[i] = b

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # [n, h, w, m]
        return self.encode(latent)


class QuantizerDecoder(nn.Module):
    def __init__(self, m: int, k: int, d: int):
        super().__init__()
        self._m = m
        self._wq = nn.Parameter(torch.empty(m, d, d))
        self._bq = nn.Parameter(torch.empty(m, d))
        self._codebook = nn.Parameter(torch.empty(m, k, d))
        self.register_buffer("_ix", torch.arange(m))

    # def decode(self, codes):
    #     n, h, w, m = codes.shape
    #     k = self._codebook.shape[1]
    #     # [n, h, w, m, k]
    #     oneHot = F.one_hot(codes.long(), k).float()
    #     # [n, c, h, w]
    #     return torch.einsum("nhwmk,mkc->nhwmc", oneHot, self._codebook).reshape(n, h, w, -1).permute(0, 3, 1, 2)

    @torch.jit.unused
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True):
        codebooks = [c for k, c in state_dict.items() if "_codebook" in k]
        if len(codebooks) != self._codebook.shape[0]:
            raise ValueError(f"Codebook shape mismatch. m in dict: {len(codebooks)}, actually: {self._codebook.shape[0]}")
        for i, c in enumerate(codebooks):
            self._codebook[i] = c
        wqs = [c for k, c in state_dict.items() if "_wq.weight" in k]
        bqs = [c for k, c in state_dict.items() if "_wq.bias" in k]
        for i, (w, b) in enumerate(zip(wqs, bqs)):
            self._wq[i] = w
            self._bq[i] = b

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [n, h, w, m]
        n, h, w, _ = codes.shape
        # use codes to index codebook (m, k, d) ==> [n, h, w, m, k] -> [n, c, h, w]
        # ix = torch.arange(self._m, device=codes.device).expand_as(codes)
        ix = self._ix.expand_as(codes)
        # [n, h, w, m, d]
        indexed = self._codebook[ix, codes]
        indexed = torch.einsum("nhwmd,mcd->nhwmc", indexed, self._wq) + self._bq
        return indexed.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        # return self.decode(codes)


class RefEncoder(nn.Module):
    def __init__(self, m, k, channel, groups, alias):
        super().__init__()
        self._levels = len(k)

        self._keys = tuple({
            "_encoder",
            "_heads",
            "_mappers",
            "_quantizers",
            "_postProcess"
        })

        self._encoder = ResidualBaseEncoder(channel, groups, alias)

        self._heads = nn.ModuleList(EncoderHead(channel, 1, alias) for _ in range(self._levels))
        self._mappers = nn.ModuleList(DownSampler(channel, 1, alias) for _ in range(self._levels - 1))
        self._quantizers = nn.ModuleList(QuantizerEncoder(m, ki, channel // m) for ki in k)
        self._deQuantizers = nn.ModuleList(QuantizerDecoder(m, ki, channel // m) for ki in k)
        # self._postProcess = nn.ModuleList(ResidualBlock(2 * channel, channel, groups=groups) for _ in range(self._levels - 1))

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        encoderDict = {k[len("_encoder."):]: v for k, v in state_dict.items() if k.startswith("_encoder.")}
        headsDict = {k[len("_heads."):]: v for k, v in state_dict.items() if k.startswith("_heads.")}
        mappersDict = {k[len("_mappers."):]: v for k, v in state_dict.items() if k.startswith("_mappers.")}
        # postProcessDict = {k[len("_postProcess."):]: v for k, v in state_dict.items() if k.startswith("_postProcess.")}
        quantizerDict = {k[len("_quantizers."):]: v for k, v in state_dict.items() if k.startswith("_quantizers.")}
        self._encoder.load_state_dict(encoderDict)
        self._heads.load_state_dict(headsDict)
        self._mappers.load_state_dict(mappersDict)
        # self._postProcess.load_state_dict(postProcessDict)
        for i, q in enumerate(self._quantizers):
            q.load_state_dict({k[len("1."):]: v for k, v in quantizerDict.items() if k.startswith(f"{i}.")})
        for i, q in enumerate(self._deQuantizers):
            q.load_state_dict({k[len("1."):]: v for k, v in quantizerDict.items() if k.startswith(f"{i}.")})

    @property
    @torch.jit.unused
    def keys(self):
        return self._keys

    def forward(self, x: torch.Tensor, cAndPadding: torch.Tensor) -> Tuple[List[torch.LongTensor], torch.Tensor]:
        codes = list()
        latent = self._encoder(x)
        for head, mapper, quantizer, deQuantizer in zip(self._heads, self._mappers, self._quantizers, self._deQuantizers):
            z = head(latent)
            latent = mapper(latent)
            code = quantizer(z)
            hard = deQuantizer(code)
            latent = latent - hard # postProcess(torch.cat((latent, hard), 1))
            codes.append(code)
        z = self._heads[-1](latent)
        codes.append(self._quantizers[-1](z))
        # codes from small to big
        return codes, cAndPadding


class RefDecoder(nn.Module):
    def __init__(self, m, k, channel, alias):
        super().__init__()
        self._levels = len(k)

        self._keys = tuple({
            "_decoder",
            "_reverses",
            "_scatters",
            "_quantizers",
            "_finalProcess"
        })

        self._decoder = ResidualBaseDecoder(channel, 1)

        self._reverses0 = UpSampler(channel, 1, alias)
        self._reverses = nn.ModuleList(UpSampler(channel, 1, alias) for _ in range(self._levels - 1))
        self._scatters = nn.ModuleList(Director(channel, 1, alias) for _ in range(self._levels - 1))
        # self._finalProcess = nn.ModuleList(ResidualBlock(2 * channel, channel, 1) for _ in range(self._levels - 1))
        ################### REVERSE THE QUANTIZER ###################
        self._quantizers0 = QuantizerDecoder(m, k[-1], channel // m)
        self._quantizers = nn.ModuleList(QuantizerDecoder(m, ki, channel // m) for ki in k[-2::-1])

    @torch.jit.unused
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        decoderDict = {k[len("_decoder."):]: v for k, v in state_dict.items() if k.startswith("_decoder.")}
        self._decoder.load_state_dict(decoderDict)
        ############################### IMPORTANT: LOAD IN REVERSE ORDER ###############################
        reversesDict = {k[len("_reverses."):]: v for k, v in state_dict.items() if k.startswith("_reverses.")}
        scattersDict = {k[len("_scatters."):]: v for k, v in state_dict.items() if k.startswith("_scatters.")}
        # finalProcessDict = {k[len("_finalProcess."):]: v for k, v in state_dict.items() if k.startswith("_finalProcess.")}
        quantizerDict = {k[len("_quantizers."):]: v for k, v in state_dict.items() if k.startswith("_quantizers.")}

        self._reverses0.load_state_dict({k[len("1."):]: v for k, v in reversesDict.items() if k.startswith(f"{self._levels - 1}.")})
        for i, r in enumerate(self._reverses):
            r.load_state_dict({k[len("1."):]: v for k, v in reversesDict.items() if k.startswith(f"{self._levels - i - 2}.")})

        for i, s in enumerate(self._scatters):
            s.load_state_dict({k[len("1."):]: v for k, v in scattersDict.items() if k.startswith(f"{self._levels - i - 2}.")})

        # for i, s in enumerate(self._finalProcess):
            # s.load_state_dict({k[len("1."):]: v for k, v in finalProcessDict.items() if k.startswith(f"{self._levels - i - 2}.")})

        self._quantizers0.load_state_dict({k[len("1."):]: v for k, v in quantizerDict.items() if k.startswith(f"{self._levels - 1}.")})
        for i, q in enumerate(self._quantizers):
            q.load_state_dict({k[len("1."):]: v for k, v in quantizerDict.items() if k.startswith(f"{self._levels - i - 2}.")})
        ############################### IMPORTANT: LOAD IN REVERSE ORDER ###############################

    @property
    @torch.jit.unused
    def keys(self):
        return self._keys

    def forward(self, codes: List[torch.LongTensor], cAndPadding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        smallQ = self._reverses0(self._quantizers0(codes[-1]))
        for i, (scatter, quantizer, reverse) in enumerate(zip(self._scatters, self._quantizers, self._reverses)):
            code = codes[-(i + 2)]
            q = scatter(quantizer(code))

            q = smallQ + q # finalProcess(torch.cat((smallQ, q), 1))
            smallQ = reverse(q)

        return self._decoder(smallQ).clamp_(-1, 1), cAndPadding



class RefEncoder5x5(RefEncoder):
    def __init__(self, m, k, channel, groups, alias):
        super(RefEncoder, self).__init__()
        self._levels = len(k)

        self._keys = tuple({
            "_encoder",
            "_heads",
            "_mappers",
            "_quantizers",
            "_postProcess"
        })

        self._encoder = BaseEncoder5x5(channel, groups, alias)

        self._heads = nn.ModuleList(EncoderHead5x5(channel, 1, alias) for _ in range(self._levels))
        self._mappers = nn.ModuleList(DownSampler5x5(channel, 1, alias) for _ in range(self._levels - 1))
        self._quantizers = nn.ModuleList(QuantizerEncoder(m, ki, channel // m) for ki in k)
        self._deQuantizers = nn.ModuleList(QuantizerDecoder(m, ki, channel // m) for ki in k)
        # self._postProcess = nn.ModuleList(ResidualBlock(2 * channel, channel, groups=groups) for _ in range(self._levels - 1))


class RefDecoder5x5(RefDecoder):
    def __init__(self, m, k, channel, alias):
        super(RefDecoder, self).__init__()
        self._levels = len(k)

        self._keys = tuple({
            "_decoder",
            "_reverses",
            "_scatters",
            "_quantizers",
            "_finalProcess"
        })

        self._decoder = BaseDecoder5x5(channel, 1)

        self._reverses0 = UpSampler5x5(channel, 1, alias)
        self._reverses = nn.ModuleList(UpSampler5x5(channel, 1, alias) for _ in range(self._levels - 1))
        self._scatters = nn.ModuleList(Director5x5(channel, 1, alias) for _ in range(self._levels - 1))
        # self._finalProcess = nn.ModuleList(ResidualBlock(2 * channel, channel, 1) for _ in range(self._levels - 1))
        ################### REVERSE THE QUANTIZER ###################
        self._quantizers0 = QuantizerDecoder(m, k[-1], channel // m)
        self._quantizers = nn.ModuleList(QuantizerDecoder(m, ki, channel // m) for ki in k[-2::-1])
