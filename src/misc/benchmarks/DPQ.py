import torch
from torch import nn

from mcqc.metrics.PairwiseMetrics import L2DistanceWithNorm

from Runner import Run

class DPQ(nn.Module):
    def __init__(self, M, K, D):
        super().__init__()
        self._m = M
        self._k = K
        self._d = D
        self.codebook = nn.ParameterList([nn.Parameter(torch.randn(K, D)) for _ in range(M)])

    def Codebook(self):
        return torch.stack(list(self.codebook.parameters()), 0)

    def forward(self, x):
        hardCodes = list()
        residual = x.clone()
        softs = list()
        hards = list()
        qSoft = list()
        qHard = list()
        for i in range(self._m):
            codebooki = self.codebook[i]
            distance = -L2DistanceWithNorm(residual.clone(), codebooki)
            soft = torch.softmax(distance, -1) @ codebooki
            hardCode = distance.argmax(1)
            hardCodes.append(hardCode.detach())
            oneHot = nn.functional.one_hot(hardCode, self._k)
            hard = oneHot.float() @ codebooki
            residual -= hard
            qSoft.append(soft)
            qHard.append(hard)
            softs.append(nn.functional.mse_loss(x, sum(qSoft)))
            hards.append(nn.functional.mse_loss(x, sum(qHard)))

        hardCodes = torch.stack(hardCodes, -1)
        softDistortion = sum(softs)
        hardDistortion = sum(hards)
        jointCenter = nn.functional.mse_loss(sum(qSoft), sum(qHard))
        loss = 0.1 * softDistortion + hardDistortion + 0.1 * jointCenter
        return hardCodes, loss

if __name__ == "__main__":
    model = DPQ(16, 256, 512).cuda()
    Run(model, lr=.05)
