import logging
from typing import Any, List, Tuple, Union, Optional
import os

from rich.progress import BarColumn, Progress, TimeElapsedColumn
import torch
from torch import nn
import torch.distributed as dist
import random
import numpy as np
from vlutils.saver import Saver, DummySaver, StrPath


def initializeBaseConfigs(port: str, rank: int, worldSize: int, logger = logging):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    logger.debug("DDP master addr: `%s`", "127.0.0.1")
    logger.debug("DDP master port: `%s`", port)
    torch.autograd.set_detect_anomaly(False) # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore
    logger.debug("Autograd detect anomaly = `%s`", False)
    logger.debug("         CuDNN bechmark = `%s`", True)
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    logger.debug("            Random seed = `%d`", 3407)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)
    logger.debug("Process group = `%s`, world size = `%d`", "NCCL", worldSize)


def getRichProgress(disable: bool = False):
    return Progress("[[i blue]{task.description}[/]]: [progress.percentage]{task.fields[progress]}", BarColumn(), TimeElapsedColumn(), "{task.fields[suffix]}", transient=True, disable=disable, expand=True)


def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: str = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: str = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)


class DiffEMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._diffShadow: torch.Tensor
        self._shadow: torch.Tensor
        self._previous: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)
        self.register_buffer("_diffShadow", torch.empty(size) * torch.nan)
        self.register_buffer("_previous", torch.empty(size) * torch.nan)

    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._diffShadow)):
            self._previous.copy_(x)
            return self._shadow.zero_(), self._diffShadow.zero_().add_(1)
        self._shadow -= self._decay * (self._shadow - x)
        diff = x - self._previous
        self._diffShadow -= self._decay * (self._diffShadow - diff)
        self._previous.copy_(x)
        return self._shadow, self._diffShadow


def nop(*_, **__):
    pass
