"""Package of datasets.

Exports:
    Basic: A Basic dataset that reads all images from a directory.
    BasicLMDB: A Basic dataset that reads from a LMDB.
"""
from typing import Union
import logging
import os
import json
import glob
from torch.utils.data import DataLoader, DistributedSampler
from vlutils.logger import LoggerBase
from vlutils.saver import StrPath
import torch

from .transforms import getTrainingPreprocess, getEvalTransform
from .dataset import Basic, BasicLMDB

from torchvision.io.image import ImageReadMode, decode_image
import webdataset as wds

__all__ = [
    "Basic",
    "BasicLMDB",
]


class DummyLoader(DataLoader):
    def __init__(self, **_):
        return


def wdsDecode(sample):

    sample = torch.ByteTensor(torch.ByteStorage.from_buffer(bytearray(sample['jpg'])))
    # UNCHANGED --- Slightly speedup
    # No need to force RGB. Transforms will handle it.
    sample = decode_image(sample, ImageReadMode.UNCHANGED)
    if len(sample.shape) < 3:
        sample = sample.expand(3, *sample.shape)
    if sample.shape[0] == 1:
        sample = sample.repeat((3, 1, 1))
    elif sample.shape[0] == 4:
        sample = sample[:3]
    return sample

def getTrainLoader(rank: int, worldSize: int, datasetPath: StrPath, batchSize: int, logger: Union[logging.Logger, LoggerBase] = logging.root):
    allTarGZ = glob.glob(os.path.join(datasetPath, '*.tar.gz'))
    # NOTE: no need to use disbtribued sampler, since shuffle have difference RNG over time and pid.
    # NOTE: do not call .repeat(), it hangs!
    # NOTE: if number of shard < nodes, do not use shard_splitter, it hangs!
    trainDataset = wds.WebDataset(allTarGZ).shuffle(1000).map(wdsDecode).map(getTrainingPreprocess())
    # trainDataset = BasicLMDB(datasetPath, transform=getTrainingPreprocess())
    logger.debug("Create training set: %s", trainDataset)
    # trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    # trainLoader = wds.WebLoader(trainDataset, batch_size=batchSize, num_workers=0, pin_memory=True, prefetch_factor=None, persistent_workers=False)
    # NOTE: we use native dataloader, num_worker=2 is very enough
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return trainLoader

def getValLoader(datasetPath: StrPath, disable: bool = False, logger: Union[logging.Logger, LoggerBase] = logging.root):
    if disable:
        return DummyLoader()
    valDataset = Basic(datasetPath, transform=getEvalTransform())
    logger.debug("Create validation set: %s", valDataset)
    return DataLoader(valDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
