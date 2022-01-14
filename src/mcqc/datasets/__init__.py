"""Package of datasets.

Exports:
    Basic: A Basic dataset that reads all images from a directory.
    BasicLMDB: A Basic dataset that reads from a LMDB.
    Prefetcher: A DataLoader wrapper that prefetches data for speed-up.
"""
import os
from torch.utils.data import DataLoader, DistributedSampler
from vlutils.saver import StrPath

from .transforms import getTrainingFullTransform, getTrainingTransform, getTrainingPreprocess, getEvalTransform, getTestTransform
from .dataset import Basic, BasicLMDB
from .prefetcher import Prefetcher


__all__ = [
    "Basic",
    "BasicLMDB",
    "Prefetcher"
]


class DummyLoader(DataLoader):
    def __init__(self, **_):
        return

def getTrainLoader(rank: int, worldSize: int, datasetName: StrPath, batchSize: int):
    trainDataset = BasicLMDB(os.path.join("data", datasetName), maxTxns=(batchSize + 4) * worldSize, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(batchSize, len(trainDataset)), num_workers=batchSize + 4, pin_memory=True, persistent_workers=True)
    return Prefetcher(trainLoader, rank, getTrainingTransform()), trainSampler

def getTrainingRefLoader(datasetName: StrPath, batchSize: int):
    trainDataset = BasicLMDB(os.path.join("data", datasetName), maxTxns=(batchSize + 4), transform=getTrainingFullTransform())
    trainLoader = DataLoader(trainDataset, batch_size=min(batchSize, len(trainDataset)), num_workers=batchSize + 4, pin_memory=True)
    return trainLoader

def getValLoader(datasetName: StrPath, batchSize: int, disable: bool = False):
    if disable:
        return DummyLoader()
    valDataset = Basic(os.path.join("data", datasetName), transform=getEvalTransform())
    return DataLoader(valDataset, batch_size=min(batchSize, len(valDataset)), shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

def getTestLoader(datasetName: StrPath, disable: bool = False):
    if disable:
        return DummyLoader()
    testDataset = Basic(os.path.join("data", datasetName), transform=getTestTransform())
    return DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
