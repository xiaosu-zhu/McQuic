import math
import torch
from torch import nn

from mcqc import Consts

__all__ = [
    "NonNegativeParametrizer",
    "LogExpMinusOne",
    "logExpMinusOne"
]


class _lowerBound(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, input_, bound):
        ctx.save_for_backward(input_, bound)
        return torch.max(input_, bound)

    @staticmethod
    def backward(ctx, grad_output):
        input_, bound = ctx.saved_tensors
        pass_through_if = (input_ >= bound) | (grad_output < 0)
        return pass_through_if.type(grad_output.dtype) * grad_output, None

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    def __init__(self, bound: float):
        """Lower bound operator.

        Args:
            bound (float): The lower bound.
        """
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused # type: ignore
    def lower_bound(self, x):
        return _lowerBound.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting(): # type: ignore
            return torch.max(x, self.bound) # type: ignore
        return self.lower_bound(x)



class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.
    Used for stability during training.
    """
    def __init__(self, minimum: float = 0.0, eps: float = Consts.Eps):
        """Non negative reparametrization.

        Args:
            minimum (float, optional): The lower bound. Defaults to 0.
            reparam_offset (float, optional): Eps for stable training. Defaults to 2**-18.
        """
        super().__init__()

        minimum = float(minimum)
        eps = float(eps)

        self.register_buffer("eps", torch.Tensor([eps ** 2]))
        bound = (minimum + eps ** 2) ** 0.5
        self.lowerBound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.eps, self.eps)) # type: ignore

    def forward(self, x):
        out = self.lowerBound(x)
        out = out ** 2 - self.eps
        return out


class _logExpMinusOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return (x.exp() - 1 + Consts.Eps).log()

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        passThroughIf = x > bound
        remaining = ~passThroughIf
        return passThroughIf * grad_output + remaining * grad_output * x.exp() / (x.exp() - 1 + Consts.Eps), None

class LogExpMinusOne(nn.Module):
    def __init__(self, eps: float = Consts.Eps) -> None:
        super().__init__()
        eps: torch.Tensor = torch.tensor(eps, dtype=torch.float)
        self.register_buffer("_bound", ((1 + eps) / eps).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _logExpMinusOne.apply(x, self._bound)

def logExpMinusOne(x: torch.Tensor, eps: float = Consts.Eps) -> torch.Tensor:
    eps: torch.Tensor = torch.tensor(eps, dtype=torch.float, device=x.device)
    return _logExpMinusOne.apply(x, ((1 + eps) / eps).log())
