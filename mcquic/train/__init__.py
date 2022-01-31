from shutil import copy2
from typing import Tuple, Union
import os
from numpy import save

import torch
from torch import nn
import torch.optim
from vlutils.config import summary

from mcquic import Config, Consts
from mcquic.models.compressor import BaseCompressor, Compressor
from mcquic.loss import CompressionLossBig
from mcquic.utils.helper import getSaver, initializeBaseConfigs
from mcquic.datasets import getTrainLoader, getTestLoader, getValLoader
from mcquic.utils.registry import OptimizerRegistry, ValueTunerRegistry, LrSchedulerRegistry

from .trainer import getTrainer
from .lrSchedulers import *
from .valueTuners import *


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
        tmpFile = copy2(resume, os.path.join(Consts.TempDir, "resume.ckpt"), follow_symlinks=False)
        # stateDict = torch.load(resume, {"cuda:0": f"cuda:{rank}"})[Consts.Fingerprint]
    else:
        tmpFile = None

    saver = getSaver(saveDir, saveName="saved.ckpt", loggerName=Consts.Name, loggingLevel="DEBUG" if debug else "INFO", config=config, reserve=False, disable=rank != 0, dumpFile=Consts.RootDir)

    saver.info("Here is the whole config during this run: \r\n%s", summary(config))

    saver.debug("Creating the world···")

    initializeBaseConfigs(port, rank, worldSize, logger=saver)
    saver.debug("Base configs initialized.")

    optimizerFn = OptimizerRegistry.get("Lamb", logger=saver)
    schdrFn = LrSchedulerRegistry.get(config.Schdr.type, logger=saver)
    valueTunerFns = [ValueTunerRegistry.get(config.RegSchdr.type, saver), ValueTunerRegistry.get(config.TempSchdr.type, logger=saver)]

    trainer = getTrainer(rank, config, lambda: modelFn(config.Model.channel, config.Model.m, config.Model.k, config.Model.target), optimizerFn, schdrFn, valueTunerFns, saver)

    if tmpFile is not None:
        saver.info("Found ckpt to resume at %s", resume)
        trainer.restoreStates(tmpFile)

    trainLoader, trainSampler = getTrainLoader(rank, worldSize, config.Dataset, config.BatchSize, logger=saver)
    valLoader = getValLoader(config.ValDataset, config.BatchSize, disable=rank != 0, logger=saver)
    testLoader = getTestLoader(config.ValDataset, disable=rank != 0, logger=saver)
    saver.debug("Train, validation and test datasets mounted.")

    trainer.train(trainLoader, trainSampler, valLoader, testLoader)

    saver.debug(summary(config))
    saver.info("Bye.")
