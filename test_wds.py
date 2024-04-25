import webdataset as wds
import torch
from torchvision.io.image import ImageReadMode, decode_image
import webdataset as wds
import glob
import os
from torch.utils.data import DataLoader

from typing import Union

from mcquic.data.transforms import getTrainingPreprocess

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



def getTrainLoader(datasetPath, batchSize: int):
    allTarGZ = glob.glob(os.path.join(datasetPath, '*.tar.gz'))
    # NOTE: no need to use disbtribued sampler, since shuffle have difference RNG over time and pid.
    trainDataset = wds.WebDataset(allTarGZ).shuffle(10000).map(wdsDecode).map(getTrainingPreprocess())
    # trainDataset = BasicLMDB(datasetPath, transform=getTrainingPreprocess())
    # trainSampler = DistributedSampler(trainDataset, worldSize, rank)
    # trainLoader = wds.WebLoader(trainDataset, batch_size=batchSize, num_workers=0, pin_memory=True, prefetch_factor=None, persistent_workers=False)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, num_workers=19, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return trainLoader



if __name__ == '__main__':
    loader = getTrainLoader('/mnt/nfs/CMG/mcquic/', 19)
    for e in range(20):
        for i, image in enumerate(loader):
            print(e, i)
            # if list(image.shape) != [19, 3, 512, 512]:
            #     raise RuntimeError(f'{image.shape} != [19, 3, 512, 512]')
            # print(image.shape)
