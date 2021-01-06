import torch
from torch import nn

from mcqc.layers.convs import conv3x3
from mcqc.layers.blocks import ResidualBlock, ResidualBlockUpsample, subPixelConv3x3, AttentionBlock

class Decoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            # AttentionBlock(channel),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)
