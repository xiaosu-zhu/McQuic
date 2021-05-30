import torch
import torch.nn as nn
import torch.nn.functional as F


class fineTuneFrom(torch.autograd.Function):
    """Autograd function for the `FineTuneFrom` operator."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return 0.1 * grad_output


class FineTuneFrom(nn.Module):
    """Gradient reduction operator
    """
    def forward(self, x):
        return fineTuneFrom.apply(x)
