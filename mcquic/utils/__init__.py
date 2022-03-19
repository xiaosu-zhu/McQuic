from distutils.version import StrictVersion
import warnings

from typing import List
from torch import nn
from rich import filesize

import mcquic

from .registry import *

def nop(*_, **__):
    pass


def totalParameters(model: nn.Module) -> str:
    allParams = sum(p.numel() for p in model.parameters())
    unit, suffix = filesize.pick_unit_and_suffix(allParams, ["", "k", "M", "B"], 1000)
    return f"{(allParams / unit):.4f}{suffix}"

def bppUpperBound(m: int, k: List[int], featureMapScale: List[float]):
    raise NotImplementedError


def versionCheck(versionStr: str):
    version = StrictVersion(versionStr)
    builtInVersion = StrictVersion(mcquic.__version__)

    if builtInVersion < version:
        raise ValueError(f"Version too new. Given {version}, but I'm {builtInVersion} now.")

    major, minor, revision = version.version

    bMajor, bMinor, bRev = builtInVersion.version

    if major != bMajor:
        raise ValueError(f"Major version mismatch. Given {version}, but I'm {builtInVersion} now.")

    if minor != bMinor:
        warnings.warn(f"Minor version mismatch. Given {version}, but I'm {builtInVersion} now.")
    return True
