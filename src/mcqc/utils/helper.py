from typing import Any, List, Tuple, Union
import os

from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
import torch
from torch import nn
import torch.distributed as dist
import random
import numpy as np
from vlutils.saver import Saver, DummySaver, StrPath


def initializeProcessGroup(port: str, rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.autograd.set_detect_anomaly(False) # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
    torch.manual_seed(3407)
    random.seed(3407)
    torch.cuda.set_device(rank)
    np.random.seed(3407)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)


def getRichProgress(disable: bool = False):
    return Progress("[[i blue]{task.description}[/]]: [progress.percentage]{task.fields[progress]}", BarColumn(), TimeElapsedColumn(), "{task.fields[suffix]}", transient=True, disable=disable)


def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: str = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: str = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)


class EMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._shadow: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            return self._shadow.copy_(x)
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow


def nop(*_, **__):
    pass
