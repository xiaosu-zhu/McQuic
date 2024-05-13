"""Package of datasets.

Exports:
    Basic: A Basic dataset that reads all images from a directory.
    BasicLMDB: A Basic dataset that reads from a LMDB.
"""

from typing import Union
import logging
import os
import io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import glob
from torch.utils.data import DataLoader, DistributedSampler
from vlutils.logger import LoggerBase
from vlutils.saver import StrPath
from torch.utils.data import default_collate

from mcquic.data.transforms import (
    getTrainingPreprocess,
    getEvalTransform,
    getTrainingPreprocessWithText,
)
from mcquic.data.dataset import Basic, BasicLMDB

from torchvision.transforms.functional import to_tensor
from torchvision.io.image import ImageReadMode, decode_image

import webdataset as wds
from datasets import load_dataset

__all__ = [
    "Basic",
    "BasicLMDB",
]


class DummyLoader(DataLoader):
    def __init__(self, **_):
        return


def wdsDecode(sample):
    with io.BytesIO(sample["jpg"]) as stream:
        sample = Image.open(stream).convert("RGB")
        result = to_tensor(sample.copy()).detach().clone()
        sample.close()
        # sample = torch.ByteTensor(torch.ByteStorage.from_buffer(bytearray(sample['jpg'])))
        # UNCHANGED --- Slightly speedup
        # No need to force RGB. Transforms will handle it.
        # sample = decode_image(sample, ImageReadMode.UNCHANGED)
        # if len(sample.shape) < 3:
        #     sample = sample.expand(3, *sample.shape)
        # if sample.shape[0] == 1:
        #     sample = sample.repeat((3, 1, 1))
        # elif sample.shape[0] == 4:
        #     sample = sample[:3]
    return result


def wdsDecodeWithText(sample):
    with io.BytesIO(sample["jpg"]) as stream:
        img = Image.open(stream).convert("RGB")
        result = to_tensor(img.copy()).detach().clone()
        img.close()
        # sample = torch.ByteTensor(torch.ByteStorage.from_buffer(bytearray(sample['jpg'])))
        # UNCHANGED --- Slightly speedup
        # No need to force RGB. Transforms will handle it.
        # sample = decode_image(sample, ImageReadMode.UNCHANGED)
        # if len(sample.shape) < 3:
        #     sample = sample.expand(3, *sample.shape)
        # if sample.shape[0] == 1:
        #     sample = sample.repeat((3, 1, 1))
        # elif sample.shape[0] == 4:
        #     sample = sample[:3]
    return result, sample["txt"].decode("utf-8")


def wdsImageNetWithLabel(sample):
    from mcquic.data.imagenet_classes import IMAGENET2012_CLASSES
    # with io.BytesIO(sample["jpeg"]) as stream:
    #     img = Image.open(stream).convert("RGB")
    #     image = to_tensor(img.copy()).detach().clone()
    #     img.close()

    label = IMAGENET2012_CLASSES[sample["__key__"].split("_")[0]]
    caption = f"a photo of {label}"
    image = sample["jpeg"].convert("RGB")
    
    return {"jpeg": image, "label": caption}


def getTrainLoader(
    gen: bool,
    datasetPath: StrPath,
    batchSize: int,
    logger: Union[logging.Logger, LoggerBase] = logging.root,
):
    allTarGZ = glob.glob(str(datasetPath))
    # NOTE: no need to use disbtribued sampler, since shuffle have difference RNG over time and pid.
    # NOTE: do not call .repeat(), it hangs!
    # NOTE: if number of shard < nodes, do not use shardshuffle, it hangs!
    # NOTE: !!!! DON'T set shuffle too large, it will consume very much memory !!!!
    # NOTE: they (wds) recommend to batch in advance, not in dataloader
    # NOTE: don't use their (wds) collate function, it is wrong.
    if gen:
        trainDataset = (
            load_dataset("webdataset", data_dir=datasetPath, split="train", streaming=True)
            .shuffle(seed=3407, buffer_size=10_000)
            .map(wdsImageNetWithLabel)
            .map(getTrainingPreprocessWithText())
            # wds.WebDataset(allTarGZ, shardshuffle=True, nodesplitter=wds.split_by_node)
            # .shuffle(500)
            # .map(wdsImageNetWithLabel)
            # .map(getTrainingPreprocessWithText())
            # .batched(batchSize, collation_fn=default_collate, partial=False)
        )
    else:
        trainDataset = (
            wds.WebDataset(allTarGZ, shardshuffle=True, nodesplitter=wds.split_by_node)
            .shuffle(500)
            .map(wdsDecode)
            .map(getTrainingPreprocess())
            .batched(batchSize, collation_fn=default_collate, partial=False)
        )
    logger.debug("Create training set: %s", trainDataset)
    # NOTE: we use native dataloader
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        num_workers=min(batchSize // 2, 48) if gen else min(batchSize + 4, 16),
        pin_memory=True,
        persistent_workers=False,
    )
    return trainLoader


def getValLoader(
    datasetPath: StrPath,
    disable: bool = False,
    logger: Union[logging.Logger, LoggerBase] = logging.root,
):
    if disable:
        return DummyLoader()
    valDataset = Basic(datasetPath, transform=getEvalTransform())
    logger.debug("Create validation set: %s", valDataset)
    return DataLoader(
        valDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
