import torch
from torch import nn


class RandomGamma(nn.Module):
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
    x = torch.pow(torch.max(x, torch.zeros_like(x)), randomGammas[:, None, None, None])
    return x.clamp_(0.0, 1.0)
