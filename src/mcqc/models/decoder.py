import torch
from torch import nn

from mcqc.layers.convs import conv3x3, deconv5x5
from mcqc.layers.gdn import GenDivNorm
from mcqc.layers.blocks import ResidualBlock, ResidualBlockUpsample, subPixelConv3x3, AttentionBlock, UpSample, GlobalAttentionBlock
from mcqc.layers.positional import PositionalEncoding2D


class Decoder(nn.Module):
    def __init__(self, inChannel=384, intermediateChannel=192):
        super().__init__()
        self._net = nn.Sequential(
            deconv5x5(inChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, intermediateChannel, 2),
            GenDivNorm(intermediateChannel, inverse=True),
            deconv5x5(intermediateChannel, 3, 2)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            AttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            AttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            subPixelConv3x3(channel, 3, 2),
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class ResidualGlobalDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlock(channel, channel),
            GlobalAttentionBlock(channel),
            GlobalAttentionBlock(channel),
            GlobalAttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            AttentionBlock(channel),
            ResidualBlock(channel, channel),
            ResidualBlockUpsample(channel, channel, 2),
            ResidualBlock(channel, channel),
            subPixelConv3x3(channel, 3, 2)
        )

    def forward(self, x: torch.Tensor):
        # [N, channel, H // 16, W // 16] <- [N, 3, H, W]
        return self._net(x)


class MultiScaleDecoder(nn.Module):
    def __init__(self, channel, postProcessNum, scale):
        super().__init__()
        postProcess = []
        for i in range(postProcessNum - 1):
            postProcess += [ResidualBlock(channel, channel), ResidualBlockUpsample(channel, channel, 2)]
        postProcess += [ResidualBlock(channel, channel), subPixelConv3x3(channel, 3, 2)]
        self._postProcess = nn.Sequential(*postProcess)
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


class TransformerDecoder(nn.Module):
    def __init__(self, layers: int, cin:int, rate: float = 0.1):
        super().__init__()
        self._encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(cin, 8, dropout=rate, activation="gelu"), layers)
        self._position = PositionalEncoding2D(cin, 120, 120)
        self._c = cin

    def forward(self, convZs):
        latents = list()
        for xRaw in convZs:
            n, c, h, w = xRaw.shape
            # [n, c, h, w] -> [h, w, n, c]
            encoderIn = xRaw.permute(2, 3, 0, 1)
            # [h, w, n, c] -> [h*w, n, c]
            encoderIn = self._position(encoderIn).reshape(-1, n, c)
            # encoderIn = encoderIn.reshape(-1, n, c)
            # [h*w, n, c] -> [n, c, h, w]
            x = self._encoder(encoderIn).permute(1, 2, 0).reshape(n, c, h, w)
            latents.append(x)
        return latents
