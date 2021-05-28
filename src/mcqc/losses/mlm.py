from math import log

import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence


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
        targets = targets[1:] + targets[:1]
        for logit, target in zip(logits, targets):
            # logit: [n, h*w, k] -> [n, k, h*w]
            # target: [n, h*w]
            loss = self._ceLoss(logit.permute(0, 2, 1), target)
            losses.append(loss)

        return sum(losses)


class ContextGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ceLoss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, step: int):
        losses = list()
        k = logits[0].shape[1]
        for logit, target in zip(logits, targets):
            if step % 2 == 0:
                # Discriminator step: find context relations
                # logit: [n, k, h*w]
                # target: [n, h*w]
                loss = self._ceLoss(logit, target)
            else:
                # Generator step:  corrupt context relations
                logit = logit.permute(0, 2, 1).reshape(-1, k)
                p = Categorical(logits=logit)
                q = Categorical(logits=torch.zeros_like(logit))
                loss = kl_divergence(p, q).mean()
            losses.append(loss)
        return sum(losses)
