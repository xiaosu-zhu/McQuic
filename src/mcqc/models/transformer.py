import math

import torch
from torch import nn
import torch.nn.functional as f


class MHAttention(nn.Module):
    def __init__(self, dIn, numHeads):
        super().__init__()
        self._numHeads = numHeads
        self._dIn = dIn
        self._depth = dIn // numHeads
        self._scale = math.sqrt(self._depth)
        self.wq = nn.Linear(dIn, dIn)
        self.wk = nn.Linear(dIn, dIn)
        self.wv = nn.Linear(dIn, dIn)
        self.pl = nn.Linear(dIn, dIn)

    def _splitHeads(self, x, n):
        return x.reshape(n, -1, self._numHeads, self._depth).permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # [n, nHead, seq, depth]
        q = self._splitHeads(q, len(q))
        k = self._splitHeads(k, len(k))
        v = self._splitHeads(v, len(v))

        attention = self._scaledDotProductAttention(q, k, v, self._scale, mask)
        # [n, seq, dIn]
        attention = attention.permute(0, 2, 1, 3).reshape(len(q), -1, self._dIn)
        return self.pl(attention)

    @staticmethod
    def _scaledDotProductAttention(q, k, v, scale, mask):
        """计算注意力权重。
        q, k, v 必须具有匹配的前置维度。
        k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
        虽然 mask 根据其类型（填充或前瞻）有不同的形状，
        但是 mask 必须能进行广播转换以便求和。

        参数:
            q: 请求的形状 == (N or 1, ..., seq_len_q, depth)
            k: 主键的形状 == (N or 1, ..., seq_len_k, depth)
            v: 数值的形状 == (N or 1, ..., seq_len_k, depth_v)
            mask: Float 张量，其形状能转换成
                        (N or 1, ..., seq_len_q, seq_len_k)。默认为None。

        返回值:
            输出 (N, ..., seq_len_q, depth_v)，注意力权重
        """
        # [N, ..., lq, lk]
        qkLogits = torch.matmul(q, k.transpose(-1, -2)) / scale

        # 将 mask 加入到缩放的张量上。
        if mask is not None:
            qkLogits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
        # 相加等于1。
        weights = torch.softmax(qkLogits, -1)    # (N, ..., seq_len_q, seq_len_k)

        # [N, ..., lq, depthV]
        return torch.matmul(weights, v)


class EncoderLayer(nn.Module):
    def __init__(self, dIn, numHeads, dHidden, rate=0.1):
        super().__init__()
        self._mha = MHAttention(dIn, numHeads)
        self._ffn = nn.Sequential(nn.Linear(dIn, dHidden), nn.GELU(), nn.Linear(dHidden, dIn))

        self._ln1 = nn.LayerNorm(dIn)
        self._ln2 = nn.LayerNorm(dIn)

        self._dropout1 = nn.Dropout(rate, False)
        self._dropout2 = nn.Dropout(rate, False)

    def forward(self, inputs):
        x, mask = inputs
        attended = self._dropout1(self._mha(x, x, x, mask))
        out1 = self._ln1(attended + x)
        ffnOut = self._dropout2(self._ffn(out1))
        return (self._ln2(out1 + ffnOut), mask)


class DecoderLayer(nn.Module):
    def __init__(self, dIn, numHeads, dHidden, rate=0.1):
        super().__init__()
        self._mha1 = MHAttention(dIn, numHeads)
        self._mha2 = MHAttention(dIn, numHeads)
        self._ffn = nn.Sequential(nn.Linear(dIn, dHidden), nn.GELU(), nn.Linear(dHidden, dIn))

        self._ln1 = nn.LayerNorm(dIn)
        self._ln2 = nn.LayerNorm(dIn)
        self._ln3 = nn.LayerNorm(dIn)

        self._dropout1 = nn.Dropout(rate, False)
        self._dropout2 = nn.Dropout(rate, False)
        self._dropout3 = nn.Dropout(rate, False)

    def forward(self, inputs):
        x, y, xMask, yMask = inputs
        attendedX = self._dropout1(self._mha1(x, x, x, xMask))
        outX = self._ln1(attendedX + x)
        attendedXY = self._dropout2(self._mha2(outX, y, y, yMask))
        outXY = self._ln2(attendedXY + outX)
        ffnOut = self._dropout2(self._ffn(outXY))
        return (self._ln3(outXY + ffnOut), y, xMask, yMask)


class Encoder(nn.Module):
    def __init__(self, numLayers, dIn, numHeads, dHidden, rate=0.1):
        super().__init__()
        self._layers = nn.Sequential(*[EncoderLayer(dIn, numHeads, dHidden, rate) for _ in range(numLayers)])
        self._dropout = nn.Dropout(rate, False)

    def forward(self, x, mask=None):
        x, _ = self._layers((self._dropout(x), mask))
        return x


class Decoder(nn.Module):
    def __init__(self, numLayers, dIn, numHeads, dHidden, rate=0.1):
        super().__init__()
        self._layers = nn.Sequential(*[DecoderLayer(dIn, numHeads, dHidden, rate) for _ in range(numLayers)])
        self._dropout = nn.Dropout(rate, False)

    def forward(self, x, y, xMask=None, yMask=None):
        x, _, _, _ = self._layers((self._dropout(x), y, xMask, yMask))
        return x


class Transformer(nn.Module):
    def __init__(self, numLayers, dIn, numHeads, dHidden, rate=0.1):
        super().__init__()
        self._encoder = Encoder(numLayers, dIn, numHeads, dHidden, rate)
        self._decoder = Decoder(numLayers, dIn, numHeads, dHidden, rate)

    def forward(self, x, y, xMask=None, yMask=None):
        '''
            x: [N or 1, ..., T1, D]
            y: [N or 1, ..., T2, D]
            return: [N or 1, ..., T1, D]
        '''
        m = x.shape
        x = self._encoder(x, xMask)
        x = self._decoder(x, y, xMask, yMask)
        return x
