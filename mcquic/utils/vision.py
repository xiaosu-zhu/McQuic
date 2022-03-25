import random
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.distributions import Categorical


__all__ = [
    "RandomGamma",
    "DeTransform",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomAutocontrast",
    "Masking"
]


class BatchRandomGamma(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        randomChoice = torch.randint(0, 4, (len(x), ))
        x[randomChoice == 0] = srgbToLinear(x[randomChoice == 0])
        x[randomChoice == 1] = linearToSrgb(x[randomChoice == 1])
        x[randomChoice == 2] = randomGamma(x[randomChoice == 2], torch.rand((x[randomChoice == 2].shape[0], ), device=x.device) * 1.95 + 0.05)
        return x

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
        identity
    ]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 3)
        return self._fns[choice](x)


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
        tensor[flipped] = F.hflip(tensor[flipped])
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
        tensor[flipped] = F.vflip(tensor[flipped])
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
        img[picked] = F.autocontrast(img[picked])
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
        mask = self._categorical.sample((n, 1, h, w)).byte().to(images.device) # type: ignore
        return (mask == 0) * images + (mask == 1) * zeros
