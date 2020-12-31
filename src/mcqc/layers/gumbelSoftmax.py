"""Module of GumbelSoftmax layer"""
import torch
import torch.nn.functional as F
from torch import nn
from deprecated import deprecated

@deprecated("Not used.")
class GumbelSoftmax(nn.Module):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax

    Explanation http://amid.fish/assets/gumbel.html

    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)

    Papers:
        [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
        [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def __init__(self, temperature, eps=1e-16):
        super().__init__()
        self._temperature = temperature
        self._eps = eps

    @staticmethod
    def _sampleGumbel(sampleShape, eps, device):
        p = torch.rand(sampleShape).to(device)
        p[p == 0.0] = eps
        return -torch.log(-torch.log(p))

    @staticmethod
    def gumbel_softmax(logits, temperature, eps, hard):
        transformed = logits + GumbelSoftmax._sampleGumbel(logits.shape, eps, logits.device)
        softmax = F.softmax(transformed / temperature, -1)
        if hard:
            return F.one_hot(torch.argmax(softmax, -1), softmax.shape[-1]) - softmax.detach() + softmax
        return softmax

    def forward(self, logits, samples, soft):
        if samples is None:
            return self.gumbel_softmax(logits, self._temperature, self._eps, hard=True)
        else:
            return -torch.sum(-samples * F.log_softmax(logits, -1), -1)
