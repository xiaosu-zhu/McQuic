import torch
from torch import nn

from mcqc.layers.convs import conv3x3
from mcqc.layers.blocks import ResidualBlock, ResidualBlockUpsample, subPixelConv3x3, AttentionBlock, UpSample

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


class MultiScaleDecoder(nn.Module):
    def __init__(self, channel, scale):
        super().__init__()
        self._postProcess = nn.Sequential(
            # ResidualBlock(channel, channel),
            # # AttentionBlock(channel),
            # ResidualBlockUpsample(channel, channel, 2),
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
        # Upsample -> 2x, 2x
        self._scales = nn.ModuleList([UpSample(channel) for _ in range(scale)])

    def forward(self, latents):
        latents = latents[::-1]
        x = latents[0]
        for i, s in enumerate(self._scales):
            x = s(x)
            if i == len(latents) - 1:
                break
            x = x + latents[i + 1]
        return self._postProcess(x)
