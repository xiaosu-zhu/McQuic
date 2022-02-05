from .registry import *

from torch import nn
from rich import filesize

def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"
