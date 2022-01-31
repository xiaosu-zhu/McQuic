import logging
from typing import Any, List, Tuple, Union, Dict
import os

from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
from torch import nn
import torch.distributed as dist
import random
import numpy as np
from vlutils.custom import RichProgress
from vlutils.saver import Saver, DummySaver, StrPath
from vlutils.runtime import functionFullName
from vlutils.base.freqHook import FrequecyHook


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


def getRichProgress(disable: bool = False) -> RichProgress:
    return RichProgress("[i blue]{task.description}[/][b magenta]{task.fields[progress]}", TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), "{task.fields[suffix]}", refresh_per_second=6, transient=True, disable=disable, expand=True)


def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: str = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: str = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)

getSaver.__doc__ = Saver.__doc__

class EMATracker(nn.Module):
    def __init__(self, size: Union[torch.Size, List[int], Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self._shadow: torch.Tensor
        self._decay = 1 - momentum
        self.register_buffer("_shadow", torch.empty(size) * torch.nan)

    def forward(self, x: torch.Tensor):
        if torch.all(torch.isnan(self._shadow)):
            self._shadow.copy_(x)
            return self._shadow
        self._shadow -= self._decay * (self._shadow - x)
        return self._shadow


def nop(*_, **__):
    pass

def checkHook(function, name, logger=logging):
    if function is None:
        logger.debug("No \"%s\".", name)
        return nop
    fullName = functionFullName(function)
    logger.debug("\"%s\" is `%s`.", name, fullName)
    return function


class EpochFrequencyHook(FrequecyHook):
    def __call__(self, step: int, epoch: int, *args: Any, **kwArgs: Any) -> Dict[int, Any]:
        with torch.inference_mode():
            return super().__call__(epoch, step, epoch, *args, **kwArgs)
