from shutil import copy2
from typing import Tuple, Union
import os
import functools

import apex
from torch import nn
import torch.distributed as dist
from vlutils.config import summary

from mcquic import Config, Consts
from mcquic.modules.compressor import BaseCompressor, Compressor
from mcquic.loss import CompressionLossBig
from mcquic.datasets import getTrainLoader, getValLoader
from mcquic.utils.registry import OptimizerRegistry, ValueTunerRegistry, LrSchedulerRegistry

from .utils import getSaver, initializeBaseConfigs
from .trainer import getTrainer
from .lrSchedulers import *
from .valueTuners import *


OptimizerRegistry.register("Adam")(torch.optim.Adam)
OptimizerRegistry.register("Lamb")(functools.partial(apex.optimizers.FusedLAMB, set_grad_none=True))

LrSchedulerRegistry.register("ReduceLROnPlateau")(torch.optim.lr_scheduler.ReduceLROnPlateau)
LrSchedulerRegistry.register("Exponential")(torch.optim.lr_scheduler.ExponentialLR)
LrSchedulerRegistry.register("MultiStep")(torch.optim.lr_scheduler.MultiStepLR)
LrSchedulerRegistry.register("OneCycle")(torch.optim.lr_scheduler.OneCycleLR) # type: ignore


def modelFn(channel, m, k, lossTarget) -> Tuple[BaseCompressor, nn.Module]:
    compressor = Compressor(channel, m, k)
    # compressor = PQCompressorBig(config.Model.m, config.Model.k, config.Model.channel, False, False, False, False, -1)
    # print(sum([p.numel() for p in compressor.parameters()]))
    # exit()
    criterion = CompressionLossBig(lossTarget)

    return compressor, criterion

def train(rank: int, worldSize: int, port: str, config: Config, saveDir: str, resume: Union[str, None], debug: bool):
    # load ckpt before create trainer, in case it moved to other place.
    if resume is not None and os.path.exists(resume) and resume.endswith("ckpt"):
        if rank == 0:
            tmpFile = copy2(resume, os.path.join(Consts.TempDir, "resume.ckpt"), follow_symlinks=False)
        else:
            tmpFile = os.path.join(Consts.TempDir, "resume.ckpt")
    else:
        tmpFile = None


    saver = getSaver(saveDir, saveName="saved.ckpt", loggerName=Consts.Name, loggingLevel="DEBUG" if debug else "INFO", config=config, reserve=False, disable=rank != 0, dumpFile=Consts.RootDir)

    saver.info("Here is the whole config during this run: \r\n%s", summary(config))

    saver.debug("Creating the world...")

    initializeBaseConfigs(port, rank, worldSize, logger=saver)
    saver.debug("Base configs initialized.")

    dist.barrier()

    optimizerFn = OptimizerRegistry.get("Lamb", logger=saver)
    schdrFn = LrSchedulerRegistry.get(config.Schdr.type, logger=saver)
    valueTunerFns = [ValueTunerRegistry.get(config.RegSchdr.type, saver), ValueTunerRegistry.get(config.TempSchdr.type, logger=saver)]

    trainer = getTrainer(rank, config, lambda: modelFn(config.Model.channel, config.Model.m, config.Model.k, config.Model.target), optimizerFn, schdrFn, valueTunerFns, saver)

    if tmpFile is not None:
        saver.info("Found ckpt to resume at %s", resume)
        trainer.restoreStates(tmpFile)

    trainLoader, trainSampler = getTrainLoader(rank, worldSize, config.Dataset, config.BatchSize, logger=saver)
    valLoader = getValLoader(config.ValDataset, config.BatchSize, disable=rank != 0, logger=saver)
    saver.debug("Train and validation datasets mounted.")

    trainer.train(trainLoader, trainSampler, valLoader)

    saver.debug(summary(config))
    saver.info("Bye.")
