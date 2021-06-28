import torch
from torch import nn
from torch.utils.tensorboard.summary import image
from torchvision import transforms as T
from PIL import Image
from torch.distributions import Categorical
from torchvision.transforms.transforms import ToTensor

def getTrainingTransform():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def getTrainingPreprocess():
    return T.RandomCrop(512, pad_if_needed=True, padding_mode="edge")

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
