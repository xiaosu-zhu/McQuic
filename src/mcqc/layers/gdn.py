# Copyright 2020 InterDigital Communications, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/gdn.py

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
    def __init__(self, minimum: float = 0.0, eps: float = 2 ** -18):
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


class GenDivNorm(nn.Module):
    r"""Generalized Divisive Normalization layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """

    def __init__(self, inChannels: int, groups: int = 1, biasBound: float = 1e-6, weightInit: float = 0.1):
        """Generalized Divisive Normalization layer.

        Args:
            inChannels (int): Channels of input tensor.
            inverse (bool, optional): GDN or I-GDN. Defaults to False.
            beta_min (float, optional): Lower bound of beta. Defaults to 1e-6.
            gamma_init (float, optional): Initial value of gamma. Defaults to 0.1.
        """
        super().__init__()

        self._groups = groups

        biasBound = float(biasBound)
        weightInit = float(weightInit)

        self.beta_reparam = NonNegativeParametrizer(minimum=biasBound)
        beta = torch.ones(inChannels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta) # type: ignore

        self.gamma_reparam = NonNegativeParametrizer()
        # m * [cOut // m, cIn // m] -> [cOut, cIn // m]
        gamma = [weightInit * torch.eye(inChannels // self._groups) for _ in range(self._groups)]
        gamma = torch.cat(gamma, 0)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma) # type: ignore

        self._nuReparam = NonNegativeParametrizer(minimum=biasBound)
        nu = torch.ones(inChannels)
        nu = self._nuReparam.init(nu)
        self.nu = nn.Parameter(nu) # type: ignore

        self._tauReparam = NonNegativeParametrizer()
        tau = [weightInit * torch.eye(inChannels // self._groups) for _ in range(self._groups)]
        tau = torch.cat(tau, 0)
        tau = self._tauReparam.init(tau)
        self.tau = nn.Parameter(tau) # type: ignore

    def forward(self, x):
        # C = x.shape[-3]
        nu = self._nuReparam(self.nu)
        tau = self._tauReparam(self.tau)

        # [C, C // groups, 1, 1]
        tau = tau[..., None, None]

        bias = F.conv2d(x, tau, nu, groups=self._groups)

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        # [C, C // groups, 1, 1]
        gamma = gamma[..., None, None]
        std = F.conv2d(x ** 2, gamma, beta, groups=self._groups)

        return self._normalize(x, bias, std)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x)

    def _normalize(self, x: torch.Tensor, bias: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - bias) * torch.rsqrt(std)


class InvGenDivNorm(GenDivNorm):
    r"""I-GDN layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """
    def _normalize(self, x: torch.Tensor, bias: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(std) + bias


# class EffGenDivNorm(GenDivNorm):
#     r"""Simplified GDN layer.
#     Introduced in `"Computationally Efficient Neural Image Compression"
#     <http://arxiv.org/abs/1912.08771>`_, by Johnston Nick, Elad Eban, Ariel
#     Gordon, and Johannes Ballé, (2019).
#     .. math::
#         y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}
#     """

#     def forward(self, x):
#         C = x.shape[-3]

#         beta = self.beta_reparam(self.beta)
#         gamma = self.gamma_reparam(self.gamma)
#         gamma = gamma.reshape(C, C, 1, 1)
#         norm = F.conv2d(torch.abs(x), gamma, beta)

#         if not self.inverse:
#             norm = 1.0 / norm

#         out = x * norm

#         return out
