import torch
from torch import nn
from mcqc.layers.blocks import ResidualBlockMasked
from mcqc.layers.convs import MaskedConv2d


class PixelCNN(nn.Module):
    def __init__(self, m: int, k: int, channel: int):
        super().__init__()
        self._net = nn.Sequential(
            ResidualBlockMasked(channel, channel),
            ResidualBlockMasked(channel, channel),
            MaskedConv2d(channel, m * k, bias=False, kernel_size=5, stride=1, padding=5 // 2, padding_mode="zeros")
        )
        self._m = m
        self._k = k

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        logits = self._net(x).reshape(n, self._k, self._m, h, w)
        return logits


def main():
    network = PixelCNN(2, 4, 12)
    x = torch.zeros(1, 12, 8, 8)
    x[..., 4, 4] = 1
    res = network(x)
    print(res.sum(1)[0, 0])

if __name__ == "__main__":
    main()
