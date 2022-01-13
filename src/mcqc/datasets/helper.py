import os

from torch.utils.data import DataLoader, DistributedSampler
from vlutils.saver import StrPath

from mcqc.utils.vision import getTrainingTransform, getTrainingPreprocess, getEvalTransform, getTestTransform
from .dataset import Basic, BasicLMDB
from .prefetcher import Prefetcher


class DummyLoader(DataLoader):
    def __init__(self, **_):
        return

def getTrainingSet(rank: int, worldSize: int, datasetName: StrPath, batchSize: int):
    trainDataset = BasicLMDB(os.path.join("data", datasetName), maxTxns=(batchSize + 4) * worldSize, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(batchSize, len(trainDataset)), num_workers=batchSize + 4, pin_memory=True, persistent_workers=True)
    return Prefetcher(trainLoader, rank, getTrainingTransform()), trainSampler

def getValidationSet(datasetName: StrPath, batchSize: int, disable: bool = False):
    if disable:
        return DummyLoader()
    valDataset = Basic(os.path.join("data", datasetName), transform=getEvalTransform())
    return DataLoader(valDataset, batch_size=min(batchSize, len(valDataset)), shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

def getTestSet(datasetName: StrPath, disable: bool = False):
    if disable:
        return DummyLoader()
    testDataset = Basic(os.path.join("data", datasetName), transform=getTestTransform())
    return DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
