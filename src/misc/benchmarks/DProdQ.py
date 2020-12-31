import torch
from torch import nn

from mcqc.metrics.PairwiseMetrics import L2DistanceWithNorm

from Runner import Run

class DProdQ(nn.Module):
    def __init__(self, M, K, D, R):
        super().__init__()
        self._m = M
        self._k = K
        self._d = D
        self._R = R
        self.codebook = nn.ParameterList([nn.Parameter(torch.randn(K, D // M)) for _ in range(M)])
        if self._R:
            self.rotateMatrix = nn.Parameter(torch.randn(D, D))

    def Codebook(self):
        codebook = torch.zeros([self._m, self._k, self._d], device=self.codebook[0].device)
        splitD = self._d // self._m
        for i in range(self._m):
            codebook[i, :, i * splitD:(i + 1) * splitD] = self.codebook[i]
        if self._R:
            codebook =  codebook @ torch.inverse(self.rotateMatrix)
        return codebook

    def forward(self, x):
        hardCodes = list()
        softs = list()
        hards = list()
        jointCenters = list()
        qSoft = list()
        qHard = list()
        if self._R:
            x = x @ self.rotateMatrix
        splits = torch.split(x, self._d // self._m, -1)
        for i in range(self._m):
            split = splits[i]
            codebooki = self.codebook[i]
            distance = -L2DistanceWithNorm(split, codebooki)
            soft = torch.softmax(distance, -1) @ codebooki
            hardCode = distance.argmax(1)
            hardCodes.append(hardCode.detach())
            oneHot = nn.functional.one_hot(hardCode, self._k)
            hard = oneHot.float() @ codebooki
            qSoft.append(soft)
            qHard.append(hard)
            softs.append(nn.functional.mse_loss(split, soft))
            hards.append(nn.functional.mse_loss(split, hard))
            jointCenters.append(nn.functional.mse_loss(soft, hard))

        hardCodes = torch.stack(hardCodes, -1)
        softDistortion = sum(softs)
        hardDistortion = sum(hards)
        jointCenter = sum(jointCenters)
        loss = 0.1 * softDistortion + hardDistortion + 0.1 * jointCenter
        if self._R:
            regularization = nn.functional.mse_loss(self.rotateMatrix @ self.rotateMatrix.transpose(0, 1), torch.eye(self._d, device=self.rotateMatrix.device))
            loss = loss + 0.01 * regularization
        return hardCodes, loss

if __name__ == "__main__":
    model = DProdQ(16, 256, 128, R=False).cuda()
    Run(model, .05)
