import functools
import os
import math
import random

import apex
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from absl import app
from absl import flags
from cfmUtils.runtime import queryGPU
from cfmUtils.logger import configLogging
from cfmUtils.saver import Saver
from cfmUtils.config import read, summary

from mcqc import Consts, Config
from mcqc.datasets import Basic, BasicLMDB
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.algorithms import Plain, FineTune, TwoPass, New, PixelCNN
from mcqc.models.whole import WholeAQ, WholePQBig, WholePQQ, WholePQRelax, WholeVQ, WholePQ, WholePQContext, WholePQTwoPass, WholePQNew, WholePQ5x5, WholePQPixelCNN
from mcqc.utils import getTrainingTransform, getEvalTransform, getTestTransform
from mcqc.utils.training import CosineAnnealingWarmupRestarts, CosineValue, CosineValueWithEnd, CyclicLR, CyclicValue, ExponentialValue, JumpAlter, JumpValue, MultiStepLRWithWarmUp, StepValue
from mcqc.utils.vision import getTrainingPreprocess

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("r", False, "Be careful to set to true. Whether to continue last training (with current config).")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")


def main(_):
    gpus = queryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.VRam + 256) if config.VRam > 0 else -1, writeOSEnv=True)
    worldSize = len(gpus)
    _changeConfig(config, worldSize)
    mp.spawn(train, (worldSize, config, saveDir, FLAGS.r, FLAGS.debug), worldSize)

def _changeConfig(config: Config, worldSize: int):
    batchSize = config.BatchSize * worldSize
    exponent = math.log2(batchSize)
    scale = 3 - exponent / 2
    if "lr" in config.Optim.params:
        config.Optim.params["lr"] /= (2 ** scale)

def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    # if "MASTER_PORT" not in os.environ:
    #     os.environ["MASTER_PORT"] = str(random.randrange(10000, 65536))
    os.environ["MASTER_PORT"] = "19936"
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(3407)
    random.seed(3407)
    torch.cuda.set_device(rank)
    np.random.seed(3407)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)
    # dist.barrier(device_ids=[rank])


def train(rank: int, worldSize: int, config: Config, saveDir: str, continueTrain: bool, debug: bool):
    _generalConfig(rank, worldSize)
    savePath = Saver.composePath(saveDir, "saved.ckpt")
    if rank == 0:
        saver = Saver(saveDir, "saved.ckpt", config, reserve=continueTrain)
        logger = configLogging(saver.SaveDir, Consts.LoggerName, "DEBUG" if debug else "INFO", rotateLogs=-1)
        logger.info("\r\n%s", summary(config))
    else:
        saver = None
        logger = None
    model = createRawModel(**kwArgs)
    trainer = Trainer(**kwArgs)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # def optimWrapper(lr, params, weight_decay):
    #     return torch.optim.AdamW(params, lr, amsgrad=True, eps=Consts.Eps, weight_decay=weight_decay)
    # def schdrWrapper(optim):
    #     return torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)

    trainDataset = BasicLMDB(os.path.join("data", config.Dataset), maxTxns=(config.BatchSize + 4) * worldSize, repeat=config.Repeat, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)

    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(config.BatchSize, len(trainDataset)), num_workers=config.BatchSize + 4, pin_memory=True, drop_last=False, persistent_workers=True)
    prefetcher = Prefetcher(trainLoader, rank, getTrainingTransform())
    trainer.run(prefetcher, trainSampler)




class Trainer:
    def __init__(self, model):
        super().__init__()
        self._rank = dist.get_rank()
        self._worldSize = dist.get_world_size()
        torch.cuda.set_device(self._rank)

        self._model = DistributedDataParallel(model.to(self._rank), device_ids=[self._rank], output_device=self._rank, broadcast_buffers=False)


    # pylint: disable=too-many-locals,arguments-differ
    def run(self, trainLoader: Prefetcher, sampler: DistributedSampler, evalLoader: DataLoader, testLoader: DataLoader):
        step = 0

        for i in range(epochs):
            sampler.set_epoch(i)
            for images in tqdm(trainLoader, ncols=40, bar_format="Epoch [%3d] {n_fmt}/{total_fmt} |{bar}|" % (i + 1), total=totalBatches, leave=False, disable=self._rank != 0):
                self._optimizer.zero_grad()
                loss = self._model(input)
                loss.backward()
                # if True:
                #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
                self._optimizer.step()
                step += 1

                if self._rank == 0:
                    printLog()
                    saveModel()


        if self._rank == 0:
            self._logger.info("Train finished")

if __name__ == "__main__":
    app.run(main)
