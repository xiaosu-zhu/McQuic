import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image

def getTrainingTransform():
    return T.Compose([
        # T.Resize(602, interpolation=Image.LANCZOS),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(512, pad_if_needed=True, padding_mode="reflect"),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # T.Lambda(lambda x : x + torch.randn_like(x)),
    ])

def getEvalTransform():
    return T.Compose([
        # T.Resize(602, interpolation=Image.LANCZOS),
        T.RandomCrop(512, pad_if_needed=True, padding_mode="reflect"),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # T.Lambda(lambda x : x + torch.randn_like(x)),
    ])
