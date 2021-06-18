from math import log
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from mcqc import Consts


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


class MLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ceLoss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        losses = list()
        for logit, target in zip(logits, targets):
            # logit: [n, k,...]
            # target: [n, ...]
            loss = self._ceLoss(logit, target)
            losses.append(loss)
        return sum(losses)


class ContextGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ceLoss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, step: int):
        losses = list()
        k = logits[0].shape[1]
        for logit, target in zip(logits, targets):
            if step % 2 == 0:
                # Discriminator step: find context relations
                # logit: [n, k, h*w]
                # target: [n, h*w]
                loss = torch.maximum(self._ceLoss(logit, target), torch.ones_like(target, dtype=torch.float32) * 0.02).mean()
            else:
                # Generator step: corrupt context relations
                # logit = logit.permute(0, 2, 1).reshape(-1, k)
                # p = Categorical(logits=logit)
                # q = Categorical(logits=torch.zeros_like(logit))
                # loss = kl_divergence(p, q).mean()
                loss = -torch.minimum(self._ceLoss(logit, target), -torch.log(torch.ones_like(target, dtype=torch.float32) / k)).mean()
            losses.append(loss)
        return sum(losses) / len(losses)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, movingMean: torch.Tensor):
        ctx.save_for_backward(x, movingMean)
        return x.exp().mean().log()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, movingMean = ctx.saved_tensors
        grad = grad_output * x.exp() / (movingMean + Consts.Eps) / x.shape[0]
        return grad, None


class InfoMaxLoss(nn.Module):
    """ Mutual Information Neural Estimation (MINE) and minimization by two-player MiniMax game """
    def __init__(self, momentum=0.99):
        super().__init__()
        self.register_buffer("_ema", torch.zeros([]))
        self._alpha = 1 - momentum
        # self._loss = nn.BCEWithLogitsLoss()

    def forward(self, logitsCondition: torch.Tensor, logitsJoint: torch.Tensor, step: int):
        expMean = (logitsJoint.detach().logsumexp(0) - log(len(logitsJoint)))
        self._ema -= self._alpha * (self._ema - expMean)
        loss = EMALoss.apply(logitsJoint, self._ema)
        # dLoss = self._loss(logitsCondition, torch.ones_like(logitsCondition)) + self._loss(logitsJoint, torch.zeros_like(logitsJoint))
        return (-logitsCondition.mean()) + loss
