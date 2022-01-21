import torch
from torchvision import transforms as T

from mcquic.utils.vision import RandomHorizontalFlip, RandomVerticalFlip, RandomGamma, RandomAutocontrast

def getTrainingTransform():
    return T.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAutocontrast(0.25),
        # T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        RandomGamma(),
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
