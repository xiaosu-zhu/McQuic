"""Package of datasets.

Exports:
    Basic: A Basic dataset that reads all images from a directory.
    BasicLMDB: A Basic dataset that reads from a LMDB.
"""
from typing import Union
import logging
from torch.utils.data import DataLoader, DistributedSampler
from vlutils.logger import LoggerBase
from vlutils.saver import StrPath

from .transforms import getTrainingPreprocess, getEvalTransform
from .dataset import Basic, BasicLMDB


__all__ = [
    "Basic",
    "BasicLMDB",
]


class DummyLoader(DataLoader):
    def __init__(self, **_):
        return

def getTrainLoader(rank: int, worldSize: int, datasetPath: StrPath, batchSize: int, logger: Union[logging.Logger, LoggerBase] = logging.root):
    trainDataset = BasicLMDB(datasetPath, transform=getTrainingPreprocess())
    logger.debug("Create training set: %s", trainDataset)
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    trainLoader = DataLoader(trainDataset, batch_size=min(batchSize, len(trainDataset)), sampler=trainSampler, num_workers=24, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return trainLoader, trainSampler

def getValLoader(datasetPath: StrPath, disable: bool = False, logger: Union[logging.Logger, LoggerBase] = logging.root):
    if disable:
        return DummyLoader()
    valDataset = Basic(datasetPath, transform=getEvalTransform())
    logger.debug("Create validation set: %s", valDataset)
    return DataLoader(valDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
