import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.distributions import Categorical

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


def getTrainingTransform():
    return T.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getTrainingPreprocess():
    return T.RandomCrop(512, pad_if_needed=True)

def getTrainingFullTransform():
    return T.Compose([
        T.RandomCrop(512, pad_if_needed=True),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getEvalTransform():
    return T.Compose([
        T.CenterCrop(512),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getTestTransform():
    return T.Compose([
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


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
