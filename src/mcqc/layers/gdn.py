
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "GenDivNorm",
    "EffGenDivNorm"
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
    def __init__(self, minimum: float = 0.0, reparam_offset: float = 2 ** -18):
        """Non negative reparametrization.

        Args:
            minimum (float, optional): The lower bound. Defaults to 0.
            reparam_offset (float, optional): Eps for stable training. Defaults to 2**-18.
        """
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset ** 2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal)) # type: ignore

    def forward(self, x):
        out = self.lower_bound(x)
        out = out ** 2 - self.pedestal
        return out


class GenDivNorm(nn.Module):
    r"""Generalized Divisive Normalization layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """

    def __init__(self, in_channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        """Generalized Divisive Normalization layer.

        Args:
            in_channels (int): Channels of input tensor.
            inverse (bool, optional): GDN or I-GDN. Defaults to False.
            beta_min (float, optional): Lower bound of beta. Defaults to 1e-6.
            gamma_init (float, optional): Initial value of gamma. Defaults to 0.1.
        """
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta) # type: ignore

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma) # type: ignore

    def forward(self, x):
        C = x.shape[-3]

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class EffGenDivNorm(GenDivNorm):
    r"""Simplified GDN layer.
    Introduced in `"Computationally Efficient Neural Image Compression"
    <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
    Gordon, and Johannes Ballé, (2019).
    .. math::
        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}
    """

    def forward(self, x):
        C = x.shape[-3]

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / norm

        out = x * norm

        return out
