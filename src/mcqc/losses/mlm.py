from math import log
import torch
from torch import nn


class MLMLoss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._ceLoss = nn.CrossEntropyLoss()

    def forward(self, logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        hw, n = target.shape
        positionToCalculate = mask != 0
        # [?, k]
        logit = logit[positionToCalculate]
        # [?]
        target = target[positionToCalculate]
        # noise = torch.rand(target.shape, device=target.device) < 0.012
        # randomint = torch.randint(0, self._k, noise.shape, device=noise.device)

        # target[noise] = randomint[noise]
        # [?] <-> [?] -> scalar
        loss = self._ceLoss(logit, target)

        return loss


class SAGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ceLoss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        losses = list()
        for logit, target in zip(logits, targets):
            # logit: [n, h*w, k] -> [n, k, h*w]
            # target: [n, h*w]
            loss = self._ceLoss(logit.permute(0, 2, 1), target)
            losses.append(loss)

        return sum(losses)
