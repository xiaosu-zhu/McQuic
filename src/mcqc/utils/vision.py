import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image

def getTransform():
    return T.Compose([
        T.Resize(602, interpolation=Image.LANCZOS),
        T.RandomCrop(512),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # T.Lambda(lambda x : x + torch.randn_like(x)),
    ])
