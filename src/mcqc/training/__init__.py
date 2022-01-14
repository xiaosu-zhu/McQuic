from typing import Tuple

import torch
from torch import nn
import torch.optim
from vlutils.config import summary

from mcqc import Config, Consts
from mcqc.models.compressor import BaseCompressor, Compressor
from mcqc.loss import CompressionLossBig
from mcqc.utils.helper import getSaver, initializeProcessGroup
from mcqc.datasets import getTrainingSet, getTestSet, getValidationSet
from mcqc.utils.registry import OptimizerRegistry, ValueTunerRegistry, LrSchedulerRegistry

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

def train(rank: int, worldSize: int, port: str, config: Config, saveDir: str, continueTrain: bool, debug: bool):
    saver = getSaver(saveDir, saveName="saved.ckpt", loggerName=Consts.Name, loggingLevel="DEBUG" if debug else "INFO", config=config, reserve=continueTrain, disable=rank != 0)

    saver.info(summary(config))

    saver.info("Create trainer...")

    initializeProcessGroup(port, rank, worldSize)

    optimizerFn = OptimizerRegistry.get("Lamb")
    schdrFn = LrSchedulerRegistry.get(config.Schdr.type)
    valueTunerFns = [ValueTunerRegistry.get(config.RegSchdr.type), ValueTunerRegistry.get(config.TempSchdr.type)]

    trainer = getTrainer(rank, config, lambda: modelFn(config.Model.channel, config.Model.m, config.Model.k, config.Model.target), optimizerFn, schdrFn, valueTunerFns, saver=saver)

    trainingSet, trainSampler = getTrainingSet(rank, worldSize, config.Dataset, config.BatchSize)
    validationSet = getValidationSet(config.ValDataset, config.BatchSize, disable=rank != 0)
    testSet = getTestSet(config.ValDataset, disable=rank != 0)


    if continueTrain:
        trainer.restoreStates(torch.load(saver.SavePath, {"cuda:0": f"cuda:{rank}"})[Consts.Fingerprint])
    trainer.train(trainingSet, trainSampler, validationSet, testSet)
