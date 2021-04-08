"""Module of GumbelSoftmax layer"""
import torch
import torch.nn.functional as F
from torch import nn
from deprecated import deprecated

class GumbelSoftmax(nn.Module):
    def forward(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1):
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret, y_soft
