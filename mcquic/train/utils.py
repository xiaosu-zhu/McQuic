from typing import Any, Dict, Union, Optional
import os
import logging
import random

import torch
import torch.distributed as dist
import numpy as np
from vlutils.custom import RichProgress
from vlutils.logger import LoggerBase
from vlutils.saver import Saver, DummySaver, StrPath
from vlutils.runtime import functionFullName
from vlutils.base import FrequecyHook
from rich.progress import TimeElapsedColumn, BarColumn, TimeRemainingColumn

from mcquic.utils import nop


def initializeBaseConfigs(logger: Union[logging.Logger, LoggerBase] = logging.root):
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = port
    # logger.debug("DDP master addr: `%s`", "127.0.0.1")
    # logger.debug("DDP master port: `%s`", port)
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True

    logger.debug("Autograd detect anomaly = `%s`", False)
    logger.debug("         CuDNN bechmark = `%s`", True)
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    logger.debug("            Random seed = `%d`", 3407)
    dist.init_process_group("nccl")
    # logger.debug("Process group = `%s`, world size = `%d`", "NCCL", worldSize)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def getRichProgress(disable: bool = False) -> RichProgress:
    return RichProgress("[i blue]{task.description}[/][b magenta]{task.fields[progress]}", TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), "{task.fields[suffix]}", refresh_per_second=6, transient=True, disable=disable, expand=True)


def getSaver(saveDir: StrPath, saveName: StrPath = "saved.ckpt", loggerName: str = "root", loggingLevel: Union[str, int] = "INFO", config: Any = None, autoManage: bool = True, maxItems: int = 25, reserve: bool = False, dumpFile: Optional[str] = None, activateTensorboard: bool = True, disable: bool = False):
    if disable:
        return DummySaver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)
    else:
        return Saver(saveDir, saveName, loggerName, loggingLevel, config, autoManage, maxItems, reserve, dumpFile, activateTensorboard)

getSaver.__doc__ = Saver.__doc__


def checkHook(function, name, logger: Union[logging.Logger, LoggerBase]=logging.root):
    if function is None:
        logger.debug("No <%s>.", name)
        return nop
    fullName = functionFullName(function)
    logger.debug("<%s> is `%s`.", name, fullName)
    return function


class EpochFrequencyHook(FrequecyHook):
    def __call__(self, step: int, epoch: int, *args: Any, **kwArgs: Any) -> Dict[int, Any]:
        with torch.inference_mode():
            return super().__call__(epoch, step, epoch, *args, **kwArgs)
