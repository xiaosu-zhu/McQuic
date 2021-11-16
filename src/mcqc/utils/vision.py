import torch
from torch import nn
from torchvision import transforms as T
from torch.distributions import Categorical

from mcqc.utils.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomAutocontrast, RandomChoiceAndApply

def getTrainingTransform():
    return T.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAutocontrast(0.25),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getTrainingPreprocess():
    return T.Compose([
        T.RandomCrop(512, pad_if_needed=True),
        # T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.2)], 0.15)
        T.RandomApply([T.RandomChoice([T.ColorJitter(0.4, 0, 0, 0), T.ColorJitter(0, 0.4, 0, 0), T.ColorJitter(0, 0, 0.4, 0), T.ColorJitter(0, 0, 0, 0.2), T.ColorJitter(0.4, 0.4, 0, 0), T.ColorJitter(0.4, 0, 0.4, 0), T.ColorJitter(0.4, 0, 0, 0.2), T.ColorJitter(0, 0.4, 0.4, 0), T.ColorJitter(0, 0.4, 0, 0.2), T.ColorJitter(0, 0, 0.4, 0.2), T.ColorJitter(0.4, 0.4, 0.4, 0), T.ColorJitter(0.4, 0.4, 0, 0.2), T.ColorJitter(0.4, 0, 0.4, 0.2), T.ColorJitter(0, 0.4, 0.4, 0.2), T.ColorJitter(0.4, 0.4, 0.4, 0.2)])], 0.25)
    ])

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
