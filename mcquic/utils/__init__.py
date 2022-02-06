from typing import List
from torch import nn
from rich import filesize

from .registry import *

def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"

def bppUpperBound(m: int, k: List[int], featureMapScale: List[float]):
    raise NotImplementedError
