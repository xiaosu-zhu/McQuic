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
from torch import nn
import torch.nn.functional as F

from .base import NonNegativeParametrizer

__all__ = [
    "GenDivNorm",
    "InvGenDivNorm"
]

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

    def forward(self, x):
        # C = x.shape[-3]
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        # [C, C // groups, 1, 1]
        gamma = gamma[..., None, None]
        std = F.conv2d(x ** 2, gamma, beta, groups=self._groups)

        return self._normalize(x, std)

    def _normalize(self, x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(std)


class InvGenDivNorm(GenDivNorm):
    r"""I-GDN layer.
    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).
    .. math::
       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """
    def _normalize(self, x: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(std)


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
