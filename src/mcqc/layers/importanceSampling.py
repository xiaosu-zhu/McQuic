import torch
from torch.distributions import Categorical
from torch.autograd import Function

class CategoricalImportanceSampling(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, p: Categorical, q: Categorical):
        pass
    @staticmethod
    def backward(ctx, grad_output):
        pass
