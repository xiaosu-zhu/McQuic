import os
import math
import random

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
from mcqc.algorithms import Plain, FineTune, TwoPass, New
from mcqc.models.whole import WholeAQ, WholePQRelax, WholeVQ, WholePQ, WholePQContext, WholePQTwoPass, WholePQNew
from mcqc.utils import getTrainingTransform, getEvalTransform, getTestTransform
from mcqc.utils.training import CyclicLR, CyclicValue, ExponentialValue, MultiStepLRWithWarmUp, StepValue
from mcqc.utils.vision import getTrainingPreprocess

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("r", False, "Be careful to set to true. Whether to continue last training (with current config).")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")


def main(_):
    if FLAGS.eval:
        assert FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace(), f"When --eval, --path must be set, got {FLAGS.path}."
        os.makedirs(FLAGS.path, exist_ok=True)
        saveDir = FLAGS.path
        config = read(os.path.join(saveDir, Consts.DumpConfigName), None, Config)
        # Test(config, saveDir)
    else:
        config = read(FLAGS.cfg, None, Config)
        if FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace():
            os.makedirs(FLAGS.path, exist_ok=True)
            saveDir = FLAGS.path
        else:
            saveDir = os.path.join(Consts.SaveDir, config.Dataset)
        gpus = queryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.VRam + 256) if config.VRam > 0 else -1, writeOSEnv=True)
        worldSize = len(gpus)
        _changeConfig(config, worldSize)
        mp.spawn(train, (worldSize, config, saveDir, FLAGS.r, FLAGS.debug), worldSize)

def _changeConfig(config: Config, worldSize: int):
    batchSize = config.BatchSize * worldSize
    if "lr" in config.Optim.params:
        config.Optim.params["lr"] *= math.sqrt(batchSize)

def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    # if "MASTER_PORT" not in os.environ:
    #     os.environ["MASTER_PORT"] = str(random.randrange(10000, 65536))
    os.environ["MASTER_PORT"] = "19936"
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(rank)
    random.seed(rank)
    torch.cuda.set_device(rank)
    np.random.seed(rank)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)
    # dist.barrier(device_ids=[rank])

models = {
    "Base": WholePQ,
    "Context": WholePQContext,
    "Relax": WholePQRelax,
    "TwoPass": WholePQTwoPass,
    "New": WholePQNew,
    "AQ": WholeAQ
}

methods = {
    "Plain": Plain,
    "FineTune": FineTune,
    "TwoPass": TwoPass,
     "New": New
}

optims = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD
}

schdrs = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "Exponential": torch.optim.lr_scheduler.ExponentialLR,
    "MultiStep": torch.optim.lr_scheduler.MultiStepLR,
    "MultiStepWarmUp": MultiStepLRWithWarmUp,
    "Cyclic": CyclicLR,
    "OneCycle": torch.optim.lr_scheduler.OneCycleLR
}

regSchdrs = {
    "Exponential": ExponentialValue,
    "Cyclic": CyclicValue,
    "MultiStep": StepValue
}

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
    model = models[config.Model.type](config.Model.m, config.Model.k, config.Model.channel, config.Model.withGroup, config.Model.withAtt, config.Model.withDropout, config.Model.alias, config.Model.ema)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # def optimWrapper(lr, params, weight_decay):
    #     return torch.optim.AdamW(params, lr, amsgrad=True, eps=Consts.Eps, weight_decay=weight_decay)
    # def schdrWrapper(optim):
    #     return torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)
    method = methods[config.Method](config, model, optims[config.Optim.type], schdrs.get(config.Schdr.type, None), regSchdrs.get(config.RegSchdr.type, None), saver, savePath, continueTrain, logger)

    trainDataset = BasicLMDB(os.path.join("data", config.Dataset), maxTxns=(config.BatchSize + 4) * worldSize, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)

    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(config.BatchSize, len(trainDataset)), num_workers=config.BatchSize + 4, pin_memory=True, drop_last=False, persistent_workers=True)
    prefetcher = Prefetcher(trainLoader, rank, getTrainingTransform())
    valLoader = None
    testLoader = None
    if rank == 0:
        valDataset = Basic(os.path.join("data", config.ValDataset), transform=getEvalTransform())
        testDataset = Basic(os.path.join("data", config.ValDataset), transform=getTestTransform())
        valLoader = DataLoader(valDataset, batch_size=min(config.BatchSize * 4, len(valDataset)), shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    method.run(prefetcher, trainSampler, valLoader, testLoader)


if __name__ == "__main__":
    app.run(main)
