import pathlib
from shutil import copy2
from typing import Tuple
import os

import torch
from torch import nn
import torch.distributed as dist
from vlutils.config import summary

from mcquic import Config, Consts
from mcquic.modules.compressor import BaseCompressor, Compressor
from mcquic.datasets import getTrainLoader, getValLoader
from mcquic.utils.registry import OptimizerRegistry, LrSchedulerRegistry, LossRegistry
import mcquic.train.lrSchedulers as _
import mcquic.loss

from .utils import getSaver, initializeBaseConfigs
from .trainer import getTrainer


def registerForTrain():
    try:
        import apex
        OptimizerRegistry.register("Lamb")(apex.optimizers.FusedLAMB)
    except:
        raise ImportError("`import apex` failed. Apex not installed.")
    OptimizerRegistry.register("Adam")(torch.optim.Adam)

    LrSchedulerRegistry.register("ReduceLROnPlateau")(torch.optim.lr_scheduler.ReduceLROnPlateau)
    LrSchedulerRegistry.register("Exponential")(torch.optim.lr_scheduler.ExponentialLR)
    LrSchedulerRegistry.register("MultiStep")(torch.optim.lr_scheduler.MultiStepLR)
    LrSchedulerRegistry.register("OneCycle")(torch.optim.lr_scheduler.OneCycleLR)


def modelFn(modelParams, lossTarget) -> Tuple[BaseCompressor, mcquic.loss.Distortion]:
    compressor = Compressor(**modelParams)
    criterion = LossRegistry.get(lossTarget)()

    return compressor, criterion


def ddpSpawnTraining(rank: int, worldSize: int, port: str, config: Config, saveDir: str, resume: pathlib.Path, loggingLevel: int):
    registerForTrain()


    # load ckpt before create trainer, in case it moved to other place.
    if resume is not None:
        if rank == 0:
            tmpFile = copy2(resume, os.path.join(Consts.TempDir, "resume.ckpt"), follow_symlinks=False)
        else:
            tmpFile = os.path.join(Consts.TempDir, "resume.ckpt")
    else:
        tmpFile = None


    saver = getSaver(saveDir, saveName="saved.ckpt", loggerName=Consts.Name, loggingLevel=loggingLevel, config=config.serialize(), reserve=False, disable=rank != 0)

    saver.info("Here is the whole config during this run: \r\n%s", summary(config.serialize()))

    saver.debug("Creating the world...")

    initializeBaseConfigs(port, rank, worldSize, logger=saver)
    saver.debug("Base configs initialized.")

    dist.barrier()

    optimizerFn = OptimizerRegistry.get(config.Train.Optim.Key, logger=saver)
    schdrFn = LrSchedulerRegistry.get(config.Train.Schdr.Key, logger=saver)

    trainer = getTrainer(rank, config, lambda: modelFn(config.Model.Params, config.Train.Target), optimizerFn, schdrFn, saver)

    if tmpFile is not None:
        saver.info("Found ckpt to resume at %s", resume)
        trainer.restoreStates(tmpFile)

    trainLoader, trainSampler = getTrainLoader(rank, worldSize, config.Train.TrainSet, config.Train.BatchSize, logger=saver)
    valLoader = getValLoader(config.Train.ValSet, disable=rank != 0, logger=saver)
    saver.debug("Train and validation datasets mounted.")

    trainer.train(trainLoader, trainSampler, valLoader)

    saver.debug(summary(config.serialize()))
    saver.info("Bye.")
