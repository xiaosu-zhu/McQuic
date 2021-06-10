from math import log, perm, sqrt
import torch
from torch import nn
from torch.distributions import Categorical

from dreinq.layers.layerGroup import LayerGroup
from dreinq.layers.incepBlock import IncepBlock

from mcqc.layers.positional import PositionalEncoding2D
from mcqc.models.quantizer import AttentiveQuantizer
from mcqc.layers.blocks import ResidualBlockWithStride, ResidualBlockUpsample, ResidualBlock, conv1x1


class _mlp(nn.Module):
    def __init__(self, d, hw, k, rate):
        super().__init__()
        self._ff1 = IncepBlock(d, rate)
        self._hh1 = IncepBlock(hw, rate)
        self._hh2 = LayerGroup(hw, k, rate, nn.ReLU)
        self._hh3 = IncepBlock(k, rate)
        self._ff2 = nn.Linear(d, d)
        self._position = PositionalEncoding2D(d, 120, 120, rate)
        self._k = k
        self._sqrtK = int(sqrt(self._k))

    def forward(self, x):
        n, d, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = self._position(x)
        x = x.reshape(h*w, n, d)
        # [hw, n, d]
        x = self._ff1(x) + x
        # [n, d, hw]
        x = x.permute(1, 2, 0)
        x = self._hh1(x) + x
        # [n, d, k]
        x = self._hh2(x)
        # [n, d, k]
        x = self._hh3(x) + x
        # [n, k, d]
        x = x.permute(0, 2, 1)
        x = self._ff2(x)
        # [n, d, k', k']
        return x.permute(0, 2, 1).reshape(n, d, self._sqrtK, self._sqrtK)


class MLP(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._encoder = nn.Sequential(
            ResidualBlockWithStride(d, d), # 16
            ResidualBlockWithStride(d, d), # 8
            ResidualBlockWithStride(d, d), # 4
            ResidualBlockWithStride(d, d), # 2
            # ResidualBlockWithStride(d, d), # 1
            # ResidualBlockUpsample(d, d), # 2
            ResidualBlockUpsample(d, d), # 4
            ResidualBlockUpsample(d, d), # 8
            ResidualBlockUpsample(d, d), # 16
            conv1x1(d, d)
        )
        self._decoder = nn.Sequential(
            # ResidualBlock(8, d), # 16
            ResidualBlockWithStride(d, d), # 8
            ResidualBlockWithStride(d, d), # 4
            ResidualBlockWithStride(d, d), # 2
            # ResidualBlockWithStride(d, d), # 1
            # ResidualBlockUpsample(d, d), # 2
            ResidualBlockUpsample(d, d), # 4
            ResidualBlockUpsample(d, d), # 8
            ResidualBlockUpsample(d, d), # 16
            ResidualBlockUpsample(d, d), # 32
            ResidualBlock(d, d),
            conv1x1(d, k * nHead)
        )
        # self._encoder = _mlp(d, 1024, 256, rate)
        # self._decoder = _mlp(d, 256, 1024, rate)

        # self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # self._decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        self._quantizer = AttentiveQuantizer(k, d, False, True)
        # self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._k = k
        self._sqrtK = int(sqrt(self._k))
        self._nHead = nHead


    def forward(self, latent, codes):
        # [n, d, k', k']
        z = self._encoder(latent)
        z, _, _, _ = self._quantizer(z, 1.0)
        # [n, k * m, h, w]
        decoded = self._decoder(z)
        # M * [n, k, h, w]
        return torch.chunk(decoded, self._nHead, 1), codes

    def predict(self, latent, codes):
        # [n, d, k', k']
        z = self._encoder(latent)
        z = self._quantizer.decode(self._quantizer.encode(z))
        # [n, k * m, h, w]
        decoded = self._decoder(z)

        logits = torch.chunk(decoded, self._nHead, 1)

        predicts = list()
        for l, c in zip(logits, codes):
            predicts.append(l.argmax(1) == c)
        return predicts


class EncoderDecoder(nn.Module):
    def __init__(self, d, nHead, nLayers, dFFN, k, rate=0.1):
        super().__init__()
        self._encoder = nn.Sequential(
            ResidualBlockWithStride(d, 4 * d), # 16
            ResidualBlockWithStride(4 * d, 4 * d), # 8
            ResidualBlockWithStride(4 * d, 4 * d), # 4
            ResidualBlockWithStride(4 * d, 4 * d), # 2
            # ResidualBlockWithStride(4 * d, 4 * d), # 1
            # ResidualBlockUpsample(4 * d, 4 * d), # 2
            ResidualBlockUpsample(4 * d, 4 * d), # 4
            ResidualBlockUpsample(4 * d, 4 * d), # 8
            ResidualBlockUpsample(4 * d, 4 * d), # 16
            conv1x1(4 * d, 8)
        )
        self._decoder = nn.Sequential(
            # ResidualBlock(8, 4 * d), # 16
            ResidualBlockWithStride(8, 4 * d), # 8
            ResidualBlockWithStride(4 * d, 4 * d), # 4
            ResidualBlockWithStride(4 * d, 4 * d), # 2
            # ResidualBlockWithStride(4 * d, 4 * d), # 1
            # ResidualBlockUpsample(4 * d, 4 * d), # 2
            ResidualBlockUpsample(4 * d, 4 * d), # 4
            ResidualBlockUpsample(4 * d, 4 * d), # 8
            ResidualBlockUpsample(4 * d, 4 * d), # 16
            ResidualBlockUpsample(4 * d, 4 * d), # 32
            ResidualBlock(4 * d, 4 * d),
            conv1x1(4 * d, k)
        )
        # self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # self._decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nHead, dFFN, rate, "gelu"), nLayers)
        # self._quantizer = AttentiveQuantizer(k, d, False, True)
        self._position = PositionalEncoding2D(d, 120, 120, rate)

        self._k = k
        self._sqrtK = int(sqrt(self._k))

        # self._hori1 = nn.Linear(1024, self._k)
        # self._hori2 = nn.Linear(self._k, 1024)

        # self._dropout = nn.Dropout(rate, True)
        # self._ffn = nn.Linear(d, k)

    def _createInput(self, latent: torch.Tensor):
        n, d, h, w = latent.shape
        # target = torch.zeros_like(latent)
        latent = latent.permute(2, 3, 0, 1)
        # target = target.permute(2, 3, 0, 1)
        latent = self._position(latent)
        # target = self._position(target)
        return latent.reshape(h*w, n, d) #, target.reshape(h*w, n, d)

    def forward(self, latent, code):
        # [n, 8, 16, 16]
        z = self._encoder(latent.detach())
        softZ = z.tanh()
        hardZ = z.sign()
        z = ((hardZ - softZ).detach() + softZ)
        # [n, k, 32, 32]
        logit = self._decoder(z)
        return logit, code

        # # [hw, n, d], [hw, n, d] target is agnostic to latent
        # latent = self._createInput(latent.detach())
        # hw, n, d = latent.shape
        # # [hw, n, d]
        # encoded = self._encoder(latent)
        # # [hw, n, d] -> [n, d, hw] -> [n, d, k] -> [k, n, d]
        # encoded = self._hori1(encoded.permute(1, 2, 0)).permute(2, 0, 1)
        # # [k, n, 8]
        # encoded = self._hash(encoded)

        # softZ = encoded.tanh()
        # hardZ = encoded.sign()

        # z = ((hardZ - softZ).detach() + softZ).permute(1, 2, 0).reshape(n, 8, self._sqrtK, self._sqrtK)

        # # [k, n, d] -> [n, d, k', k'] -> [n, d, k', k']
        # # z, _, _, _ = self._quantizer(encoded.permute(1, 2, 0).reshape(n, d, self._sqrtK, self._sqrtK), 1.0)

        # z = z.permute(2, 3, 0, 1)
        # z = self._position(z)
        # z = z.reshape(self._k, n, 8)

        # z = self._deHash(z)

        # # [k, n, d] -> [k, n, d]
        # decoded = self._decoder(z)
        # # [k, n, d] -> [n, d, k] -> [n, d, hw] -> [hw, n, d]
        # decoded = self._hori2(decoded.permute(1, 2, 0)).permute(2, 0, 1)

        # # [hw, n, k]
        # # predict logit
        # logit = self._ffn(self._dropout(decoded))
        # # [n, k, hw], [n, hw]
        # return logit.permute(1, 2, 0), code.reshape(n, -1)

    def predict(self, latent, code):
        # # [hw, n, d], [hw, n, d] target is agnostic to latent
        # latent = self._createInput(latent.detach())
        # hw, n, d = latent.shape
        # # [hw, n, d]
        # encoded = self._encoder(latent)
        # # [hw, n, d] -> [n, d, hw] -> [n, d, k] -> [k, n, d]
        # encoded = self._hori1(encoded.permute(1, 2, 0)).permute(2, 0, 1)
        # # [k, n, 8]
        # encoded = self._hash(encoded)

        # softZ = encoded.tanh()
        # hardZ = encoded.sign()

        # z = ((hardZ - softZ).detach() + softZ).permute(1, 2, 0).reshape(n, 8, self._sqrtK, self._sqrtK)

        # # [k, n, d] -> [n, d, k', k'] -> [n, d, k', k']
        # # z, _, _, _ = self._quantizer(encoded.permute(1, 2, 0).reshape(n, d, self._sqrtK, self._sqrtK), 1.0)

        # z = z.permute(2, 3, 0, 1)
        # z = self._position(z)
        # z = z.reshape(self._k, n, 8)

        # z = self._deHash(z)

        # # [k, n, d] -> [k, n, d]
        # decoded = self._decoder(z)
        # # [k, n, d] -> [n, d, k] -> [n, d, hw] -> [hw, n, d]
        # decoded = self._hori2(decoded.permute(1, 2, 0)).permute(2, 0, 1)

        # # [hw, n, k]
        # # predict logit
        # logit = self._ffn(self._dropout(decoded))
        # # [hw, n]
        # predict = logit.argmax(-1)
        # [n, 8, 16, 16]
        z = self._encoder(latent)
        # [n, k, 32, 32]
        logit = self._decoder(z.sign())
        predict = logit.argmax(1)
        return predict == code
