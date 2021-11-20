from typing import List
import storch
from storch.wrappers import deterministic
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from mcqc.layers.blocks import ResidualBlock
from mcqc.layers.convs import MaskedConv2d
from mcqc.layers.dropout import AQMasking, PointwiseDropout
from mcqc.models.decoder import ResidualBaseDecoder
from mcqc.models.quantizer import L2Quantizer, NonLinearQuantizer

from .encoder import Director, DownSampler, EncoderHead, ResidualAttEncoderNew, ResidualBaseEncoder, ResidualEncoder, ResidualAttEncoder, BaseEncoder5x5, Director5x5, DownSampler5x5, EncoderHead5x5
from .decoder import ResidualAttDecoderNew, ResidualDecoder, ResidualAttDecoder, UpSampler, BaseDecoder5x5, UpSampler5x5
from .contextModel import ContextModel
from .quantizer import AttentiveQuantizer, Quantizer, RelaxQuantizer


class PixelCNN(nn.Module):
    def __init__(self, m: int, k: int, channel: int):
        super().__init__()
        self._net = nn.Sequential(
            Director(channel, 1),
            ResidualBlock(channel, channel),
            MaskedConv2d(channel, m * k, bias=True, kernel_size=5, stride=1, padding=5 // 2, padding_mode="reflect")
        )
        self._m = m
        self._k = k

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        logits = self._net(x).reshape(n, self._k, self._m, h, w)
        return logits
