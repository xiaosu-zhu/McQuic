from typing import List
import storch
from storch.wrappers import deterministic
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from mcqc.layers.dropout import AQMasking, PointwiseDropout
from mcqc.models.decoder import ResidualBaseDecoder
from mcqc.models.quantizer import L2Quantizer, NonLinearQuantizer

from .encoder import Director, DownSampler, EncoderHead, ResidualAttEncoderNew, ResidualBaseEncoder, ResidualEncoder, ResidualAttEncoder
from .decoder import ResidualAttDecoderNew, ResidualDecoder, ResidualAttDecoder, UpSampler
from .contextModel import ContextModel
from .quantizer import AttentiveQuantizer, Quantizer, RelaxQuantizer

class PQCompressor(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, channel // m, withDropout, False, True, ema if ema > 0.0 else None) for _ in range(m))
        self._groupDropout = None # PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoder(channel, 1)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        fs = list()
        ts = list()
        binCounts = list()
        for quantizer, split in zip(self._quantizer, splits):
            q, c, l, (trueCode, frequency, binCount) = quantizer(split, temp)
            qs.append(q)
            codes.append(c)
            logits.append(l)
            # [n, h, w] trueCode frequency
            fs.append(frequency)
            ts.append(trueCode)
            binCounts.append(binCount)
        quantized = torch.cat(qs, 1)
        if self._groupDropout is not None:
            quantized = self._groupDropout(quantized)
        restored = torch.tanh(self._decoder(quantized))
        return restored, quantized, latent, torch.stack(ts, -1), logits


class PQCompressorBig(nn.Module):
    def __init__(self, m: int, k: List[int], channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1

        self._levels = len(k)

        self._encoder = ResidualBaseEncoder(channel, groups, alias)

        self._heads = nn.ModuleList(EncoderHead(channel, 1, alias) for _ in range(self._levels))
        self._mappers = nn.ModuleList(DownSampler(channel, 1, alias) for _ in range(self._levels - 1))
        self._quantizers = nn.ModuleList(nn.ModuleList(L2Quantizer(ki, channel // m) for _ in range(m)) for ki in k)

        self._reverses = nn.ModuleList(UpSampler(channel, 1, alias) for _ in range(self._levels))
        self._scatters = nn.ModuleList(Director(channel, 1, alias) for _ in range(self._levels - 1))

        self._groupDropout = None # PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualBaseDecoder(channel, 1)
        # self._context = ContextModel(m, k, channel)

    def prepare(self, x):
        latent = self._encoder(x)
        allOriginal = list()

        for i in range(self._levels):
            mapper = self._mappers[i] if i < self._levels - 1 else None
            head = self._heads[i]
            z = head(latent)
            if mapper is not None:
                latent = mapper(latent)
            else:
                latent = None
            c = self.encode(z, i)
            hard = self.decode(c, i)
            # n, c, h, w = hard.shape
            if latent is not None:
                latent = latent - hard
            # [n, m, h, w, c//m]
            # z = z.reshape(n, self._m, -1, h, w).permute(0, 1, 3, 4, 2)
            allOriginal.append(c)
        # list of [n, m, h, w]
        return allOriginal

    def quantize(self, latent, level, temp):
        splits = torch.chunk(latent, self._m, 1)
        codes = list()
        trueCodes = list()
        logits = list()
        hards = list()
        features = list()
        codebooks = list()
        for quantizer, split in zip(self._quantizers[level], splits):
            hard, c, tc, l, feature, codebook = quantizer(split, temp)
            codes.append(c)
            trueCodes.append(tc)
            logits.append(l)
            hards.append(hard)
            features.append(feature)
            codebooks.append(codebook)

        codes = torch.stack(codes, 1)
        trueCodes = torch.stack(trueCodes, 1)
        hards = torch.cat(hards, 1)
        logits = torch.stack(logits, 1)

        return hards, codes, trueCodes, logits, features, codebooks

    def encode(self, latent, level):
        splits = torch.chunk(latent, self._m, 1)
        codes = list()
        for quantizer, split in zip(self._quantizers[level], splits):
            c = quantizer.encode(split)
            codes.append(c)

        codes = torch.stack(codes, 1)
        return codes

    def decode(self, code, level):
        qs = list()
        for quantizer, c in zip(self._quantizers[level], code.permute(1, 0, 2, 3)):
            q = quantizer.decode(c)
            qs.append(q)

        return torch.cat(qs, 1)

    def nextLevelDown(self, x, level, temp):
        mapper = self._mappers[level] if level < self._levels - 1 else None
        head = self._heads[level]
        z = head(x)
        if mapper is not None:
            latent = mapper(x)
        else:
            latent = None
        hard, c, tc, l, features, codebooks = self.quantize(z, level, temp)
        if latent is not None:
            latent = latent - hard
        return latent, hard, c, tc, l, features, codebooks

    def deQuantize(self, q, level):
        reverse = self._reverses[level]
        return reverse(q)

    def nextLevelUp(self, q, upperQ, level):
        latent = self.deQuantize(q, level)
        if upperQ is not None:
            scatter = self._scatters[level - 1]
            return latent + scatter(upperQ)
        return latent


    def test(self, x:torch.Tensor):
        latent = self._encoder(x)

        allHards = list()
        allCodes = list()

        allHards.append(None)

        for i in range(self._levels):
            mapper = self._mappers[i] if i < self._levels - 1 else None
            head = self._heads[i]
            z = head(latent)
            if mapper is not None:
                latent = mapper(latent)
            else:
                latent = None
            c = self.encode(z, i)
            allCodes.append(c)
            hard = self.decode(c, i)
            if latent is not None:
                latent = latent - hard
            allHards.append(hard)

        quantizeds = list()
        quantizeds.extend(allHards)

        for i in range(self._levels, 0, -1):
            quantized = self.nextLevelUp(quantizeds[i], allHards[i - 1], i - 1)
            quantizeds[i - 1] = quantized

        restored = torch.tanh(self._decoder(quantizeds[0]))
        return restored, allCodes

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)

        allHards = list()
        allCodes = list()
        allTrues = list()
        allLogits = list()

        allHards.append(None)

        allFeatures = list()
        allCodebooks = list()

        for i in range(self._levels):
            latent, hards, c, tc, l, features, codebooks = self.nextLevelDown(latent, i, temp)
            allHards.append(hards)
            allCodes.append(c)
            allTrues.append(tc)
            allLogits.append(l)
            allFeatures.append(features)
            allCodebooks.append(codebooks)

        quantizeds = list()
        quantizeds.extend(allHards)

        for i in range(self._levels, 0, -1):
            quantized = self.nextLevelUp(quantizeds[i], allHards[i - 1], i - 1)
            quantizeds[i - 1] = quantized


        restored = torch.tanh(self._decoder(quantizeds[0]))
        return restored, allHards, latent, allCodes, allTrues, allLogits, allFeatures, allCodebooks

class PQCompressorQ(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias)
        self._quantizer = Quantizer(m, k, channel, False, False, True, -1)
        self._groupDropout = None # PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoder(channel, 1)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        quantized, trueCodes, logits = self._quantizer(latent, temp)
        restored = torch.tanh(self._decoder(quantized))
        return restored, quantized, latent, trueCodes, logits


class AQCompressor(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, channel, False, False, True, ema if ema > 0.0 else None) for _ in range(m))
        self._aqMask = AQMasking(0.1, True) if withDropout else None
        self._decoder = ResidualAttDecoder(channel, 1)

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        fs = list()
        ts = list()
        binCounts = list()
        for quantizer, split in zip(self._quantizer, splits):
            q, c, l, (trueCode, frequency, binCount) = quantizer(split, temp)
            qs.append(q)
            codes.append(c)
            logits.append(l)
            # [n, h, w] trueCode frequency
            fs.append(frequency)
            ts.append(trueCode)
            binCounts.append(binCount)
        # fs = sum(fs) / len(fs)
        # [M, N, C, H, W]
        quantized = torch.stack(qs, 0)
        if self._aqMask is not None:
            quantized = self._aqMask(quantized)
        quantized = quantized.sum(0)
        restored = torch.tanh(self._decoder(quantized))
        return restored, (quantized, latent), (torch.stack(codes, 1), fs, binCounts, torch.stack(ts, 1)), logits


class PQCompressorNew(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._k = k
        self._m = m
        self._encoder = ResidualAttEncoderNew(channel, self._m, self._k, alias)
        # self._groupDropout = PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoderNew(channel, self._m, self._k)

    def forward(self, x: torch.Tensor, temp: float):
        code, logit = self._encoder(x, temp)
        # if self._groupDropout is not None:
        #     quantized = self._groupDropout(quantized)
        restored = torch.tanh(self._decoder(code))
        return restored, code, logit


class PQCompressorTwoPass(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias) if withAtt else ResidualEncoder(channel, groups, alias)
        self._quantizer = Quantizer(m, k, channel, withDropout)
        self._groupDropout = PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoder(channel, groups) if withAtt else ResidualDecoder(channel, groups)

    def forward(self, x: torch.Tensor, temp: float, first: bool):
        latent = self._encoder(x)
        quantized, codes, logits = self._quantizer(latent, temp, first)
        if self._groupDropout is not None:
            quantized = self._groupDropout(quantized)
        restored = torch.tanh(self._decoder(quantized))
        return restored, (quantized, latent), codes, logits


class PQRelaxCompressor(nn.Module):
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoder(channel, groups, alias)
        self._quantizer = RelaxQuantizer(m, k, channel)
        self._groupDropout = PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoder(channel, groups)

    @deterministic
    def encode(self, x):
        return self._encoder(x)

    @deterministic(flatten_plates=True)
    def decode(self, x):
        return torch.tanh(self._decoder(x))

    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        # M * [n, c // M, h, w]
        qSamples, code, logit = self._quantizer(latent)
        qSamples: storch.Tensor = qSamples
        # if self._groupDropout is not None:
        #     qSamples = self._groupDropout(qSamples)
        restored = self.decode(qSamples)
        return restored, qSamples, code, logit


class PQContextCompressor(nn.Module):
    def __init__(self, m, k, channel, numLayers):
        super().__init__()
        self._k = k
        self._m = m
        self._encoder = ResidualEncoder(channel)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, False, True) for _ in range(m))
        self._decoder = ResidualDecoder(channel)
        self._context = nn.ModuleList(ContextModel(channel // m, 1, numLayers, channel // m, k) for _ in range(m))

    def forward(self, x: torch.Tensor, temp: float, e2e: bool):
        latent = self._encoder(x)
        # M * [n, c // M, h, w]
        splits = torch.chunk(latent, self._m, 1)
        qs = list()
        codes = list()
        logits = list()
        predicts = list()
        targets = list()
        for quantizer, context, split in zip(self._quantizer, self._context, splits):
            q, c, l, wv = quantizer(split, temp)
            predict, target = context(q, c)
            predicts.append(predict)
            targets.append(target)
            qs.append(q)
            codes.append(c)
            logits.append(l)
        quantized = torch.cat(qs, 1)
        # [n, m, h, w]
        codes = torch.stack(codes, 1)
        restored = torch.tanh(self._decoder(quantized))
        return restored, codes, logits, predicts, targets
