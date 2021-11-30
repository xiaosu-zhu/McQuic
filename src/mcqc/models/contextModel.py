from torch import nn
from mcqc.layers.convs import conv1x1
from mcqc.models.decoder import UpSampler


class ContextModel(nn.Module):
    def __init__(self, m, k, channel):
        super().__init__()
        self._net = UpSampler(channel, 1)
        self._liner = conv1x1(channel, m*k)
        self._m = m

    def forward(self, q):
        n, _, h, w = q.shape
        # [N, M*K, H, W]
        predict = self._liner(self._net(q)).reshape(n, -1, self._m, h * 2, w * 2)
        # [N, K, M, H, W]
        return predict
