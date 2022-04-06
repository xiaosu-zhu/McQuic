from typing import Tuple
import random

import torch
from torch import nn
import torch.nn.functional as tf
from torchvision.transforms import functional as F
from torch.distributions import Categorical


__all__ = [
    "RandomPlanckianJitter",
    "RandomGamma",
    "DeTransform",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomAutocontrast",
    "PatchWiseErasing"
]



# BEGIN::: From Kornia
_planckian_coeffs = torch.tensor(
    [
        [0.6743, 0.4029, 0.0013],
        [0.6281, 0.4241, 0.1665],
        [0.5919, 0.4372, 0.2513],
        [0.5623, 0.4457, 0.3154],
        [0.5376, 0.4515, 0.3672],
        [0.5163, 0.4555, 0.4103],
        [0.4979, 0.4584, 0.4468],
        [0.4816, 0.4604, 0.4782],
        [0.4672, 0.4619, 0.5053],
        [0.4542, 0.4630, 0.5289],
        [0.4426, 0.4638, 0.5497],
        [0.4320, 0.4644, 0.5681],
        [0.4223, 0.4648, 0.5844],
        [0.4135, 0.4651, 0.5990],
        [0.4054, 0.4653, 0.6121],
        [0.3980, 0.4654, 0.6239],
        [0.3911, 0.4655, 0.6346],
        [0.3847, 0.4656, 0.6444],
        [0.3787, 0.4656, 0.6532],
        [0.3732, 0.4656, 0.6613],
        [0.3680, 0.4655, 0.6688],
        [0.3632, 0.4655, 0.6756],
        [0.3586, 0.4655, 0.6820],
        [0.3544, 0.4654, 0.6878],
        [0.3503, 0.4653, 0.6933],
        [0.5829, 0.4421, 0.2288],
        [0.5510, 0.4514, 0.2948],
        [0.5246, 0.4576, 0.3488],
        [0.5021, 0.4618, 0.3941],
        [0.4826, 0.4646, 0.4325],
        [0.4654, 0.4667, 0.4654],
        [0.4502, 0.4681, 0.4938],
        [0.4364, 0.4692, 0.5186],
        [0.4240, 0.4700, 0.5403],
        [0.4127, 0.4705, 0.5594],
        [0.4023, 0.4709, 0.5763],
        [0.3928, 0.4713, 0.5914],
        [0.3839, 0.4715, 0.6049],
        [0.3757, 0.4716, 0.6171],
        [0.3681, 0.4717, 0.6281],
        [0.3609, 0.4718, 0.6380],
        [0.3543, 0.4719, 0.6472],
        [0.3480, 0.4719, 0.6555],
        [0.3421, 0.4719, 0.6631],
        [0.3365, 0.4719, 0.6702],
        [0.3313, 0.4719, 0.6766],
        [0.3263, 0.4719, 0.6826],
        [0.3217, 0.4719, 0.6882],
    ]
)

_planckian_coeffs_ratio = torch.stack(
    (
        _planckian_coeffs[:, 0] / _planckian_coeffs[:, 1],
        _planckian_coeffs[:, 2] / _planckian_coeffs[:, 1],
    ),
    1,
)

class RandomPlanckianJitter(nn.Module):
    pl: torch.Tensor
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.register_buffer('pl', _planckian_coeffs_ratio)
        self.p = p

    def forward(self, x: torch.Tensor):
        needsApply = torch.rand(x.shape[0]) < self.p
        coeffs = self.pl[torch.randint(len(self.pl), (needsApply.sum(), ))]

        r_w = coeffs[:, 0][..., None, None]
        b_w = coeffs[:, 1][..., None, None]

        willbeApplied = x[needsApply]

        willbeApplied[..., 0, :, :].mul_(r_w)
        willbeApplied[..., 2, :, :].mul_(b_w)

        return x.clamp_(0.0, 1.0)



def srgbToLinear(x: torch.Tensor):
    return torch.where(x < 0.0031308, 12.92 * x, (1.055 * torch.pow(torch.abs(x), 1 / 2.4) - 0.055))

def linearToSrgb(x: torch.Tensor):
    return torch.where(x < 0.04045, x / 12.92, torch.pow(torch.abs(x + 0.055) / 1.055, 2.4))

def randomGamma(x: torch.Tensor, randomGammas: torch.Tensor):
    x = torch.pow(x.clamp_(0), randomGammas)
    return x.clamp_(0.0, 1.0)

def identity(x: torch.Tensor):
    return x

class RandomGamma(nn.Module):
    _fns = [
        srgbToLinear,
        linearToSrgb,
        lambda x: randomGamma(x, torch.rand((), device=x.device) * 1.95 + 0.05),
        lambda x: x
    ]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return random.choice(self._fns)(x)


# END::: From Kornia


class DeTransform(nn.Module):
    _eps = 1e-3
    _maxVal = 255
    def __init__(self, minValue: float = -1.0, maxValue: float = 1.0):
        super().__init__()
        self._min = float(minValue)
        self._max = float(maxValue)

    def forward(self, x):
        x = (x - self._min) / (self._max - self._min)
        # [0, 1] to [0, 255]
        return (x * (self._maxVal + 1.0 - self._eps)).clamp(0.0, 255.0).byte()


# https://github.com/pratogab/batch-transforms
class RandomHorizontalFlip(nn.Module):
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    Args:
        p (float): probability of an image being flipped.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        flipped = torch.rand(tensor.shape[0]) < self.p
        tensor[flipped].copy_(tensor[flipped].flip(-1))
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(nn.Module):
    """Applies the :class:`~torchvision.transforms.RandomVerticalFlip` transform to a batch of images.
    Args:
        p (float): probability of an image being flipped.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        flipped = torch.rand(tensor.shape[0]) < self.p
        tensor[flipped].copy_(tensor[flipped].flip(-2))
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAutocontrast(torch.nn.Module):
    """Autocontrast the pixels of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be autocontrasted.
        Returns:
            PIL Image or Tensor: Randomly autocontrasted image.
        """
        picked = torch.rand(img.shape[0]) < self.p
        img[picked].copy_(F.autocontrast(img[picked]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


class Masking(nn.Module):
    def __init__(self):
        super().__init__()
        self._categorical = Categorical(torch.Tensor([0.85, 0.15]))

    def forward(self, images: torch.Tensor):
        n, _, h, w = images.shape
        zeros = torch.zeros_like(images)
        # [n, 1, h, w]
        mask = self._categorical.sample((n, 1, h, w)).byte().to(images.device)
        return (mask == 0) * images + (mask == 1) * zeros


class PatchWiseErasing(nn.Module):
    permutePattern: torch.Tensor
    def __init__(self, grids: Tuple[int, int] = (16, 16), p: float = 0.75) -> None:
        super().__init__()
        self.p = p
        self.grids = (1, 1, *grids)
        # [1024, ]
        permutePattern = torch.ones(grids).flatten()
        permuteAmount = round(permutePattern.numel() * p)
        permutePattern[:permuteAmount] = 0
        self.register_buffer("permutePattern", permutePattern)

    def forward(self, x: torch.Tensor):
        shape = x.shape
        h, w = shape[-2], shape[-1]
        randIdx = torch.randperm(len(self.permutePattern), device=x.device)
        self.permutePattern.copy_(self.permutePattern[randIdx])
        permutePattern = self.permutePattern.reshape(self.grids)
        eraseArea = tf.interpolate(permutePattern, (h, w))
        return x * eraseArea
