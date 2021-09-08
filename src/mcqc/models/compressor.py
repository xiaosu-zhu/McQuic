import storch
from storch.wrappers import deterministic
import torch
from torch import nn
from mcqc.layers.dropout import AQMasking, PointwiseDropout

from .encoder import ResidualAttEncoderBig, ResidualAttEncoderNew, ResidualEncoder, ResidualAttEncoder
from .decoder import ResidualAttDecoderBig, ResidualAttDecoderNew, ResidualDecoder, ResidualAttDecoder
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
    def __init__(self, m, k, channel, withGroup, withAtt, withDropout, alias, ema):
        super().__init__()
        self._k = k
        self._m = m
        if withGroup:
            groups = self._m
        else:
            groups = 1
        self._encoder = ResidualAttEncoderBig(channel, groups, alias)
        self._quantizer = nn.ModuleList(AttentiveQuantizer(k, channel // m, channel // m, withDropout, False, True, ema if ema > 0.0 else None) for _ in range(m))
        self._groupDropout = None # PointwiseDropout(0.05, True) if withDropout else None
        self._decoder = ResidualAttDecoderBig(channel, 1)

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
