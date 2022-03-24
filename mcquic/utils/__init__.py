from typing import List, Union
from pathlib import Path
from distutils.version import StrictVersion
import warnings
import hashlib

from torch import nn
from rich import filesize

import mcquic

from .registry import *

StrPath = Union[str, Path]


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


def hashOfFile(path: StrPath):
    sha256 = hashlib.sha256()

    with open(path, 'rb') as fp:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = fp.read(65536)
            if not chunk:
                break
            sha256.update(chunk)

    hashResult = sha256.hexdigest()
    return hashResult
