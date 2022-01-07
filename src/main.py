import functools
import os
import math
import random
from torch import nn
import sys

from tqdm.contrib.logging import logging_redirect_tqdm
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from absl import app
from absl import flags
from vlutils.runtime import queryGPU
from vlutils.logger import WaitingBar, configLogging
from vlutils.saver import Saver
from vlutils.config import T, read, summary

from mcqc import Consts, Config
from mcqc.datasets import Basic, BasicLMDB
from mcqc.datasets.prefetcher import Prefetcher
from mcqc.loss import CompressionLossBig
from mcqc.models.composed import Composed
from mcqc.models.compressor import Compressor, PQCompressorBig
from mcqc.training.trainer import MainTrainer, PalTrainer
from mcqc.utils import getTrainingTransform, getEvalTransform, getTestTransform
from mcqc.utils.registry import LrSchedulerRegistry, OptimizerRegistry, ValueTunerRegistry
from mcqc.utils.vision import getTrainingPreprocess

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "", "The config.json path.")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("r", False, "Be careful to set to true. Whether to continue last training (with current config).")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")

import signal

def handler(signum, frame):
    print("Please wait for process-group to clear all context...")
    # dist.barrier()
    # dist.destroy_process_group()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

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
        saver = Saver(saveDir, "saved.ckpt", "DEBUG" if debug else "INFO", config, reserve=continueTrain)
        saver.info("\r\n%s", summary(config))
    else:
        saver = None
        logger = None

    compressor = Compressor(config.Model.channel, config.Model.m, config.Model.k)
    # compressor = PQCompressorBig(config.Model.m, config.Model.k, config.Model.channel, False, False, False, False, -1)
    # print(sum([p.numel() for p in compressor.parameters()]))
    # exit()
    criterion = CompressionLossBig(config.Model.target)
    model = Composed(compressor, criterion)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # def optimWrapper(lr, params, weight_decay):
    #     return torch.optim.AdamW(params, lr, amsgrad=True, eps=Consts.Eps, weight_decay=weight_decay)
    # def schdrWrapper(optim):
    #     return torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)
    trainer = MainTrainer(config, model, OptimizerRegistry.get("Lamb"), LrSchedulerRegistry.get(config.Schdr.type), (ValueTunerRegistry.get(config.RegSchdr.type), ValueTunerRegistry.get(config.TempSchdr.type)), saver, None) if rank == 0 else PalTrainer(config, model, OptimizerRegistry.get("Lamb"), LrSchedulerRegistry.get(config.Schdr.type), (ValueTunerRegistry.get(config.RegSchdr.type), ValueTunerRegistry.get(config.TempSchdr.type)), None)

    trainDataset = BasicLMDB(os.path.join("data", config.Dataset), maxTxns=(config.BatchSize + 4) * worldSize, repeat=config.Repeat, transform=getTrainingPreprocess())
    trainSampler = DistributedSampler(trainDataset, worldSize, rank)

    trainLoader = DataLoader(trainDataset, sampler=trainSampler, batch_size=min(config.BatchSize, len(trainDataset)), num_workers=config.BatchSize + 4, pin_memory=True, drop_last=False, persistent_workers=True)
    prefetcher = Prefetcher(trainLoader, rank, getTrainingTransform())
    valLoader = None
    testLoader = None
    if rank == 0:
        valDataset = Basic(os.path.join("data", config.ValDataset), transform=getEvalTransform())
        testDataset = Basic(os.path.join("data", config.ValDataset), transform=getTestTransform())
        valLoader = DataLoader(valDataset, batch_size=min(config.BatchSize, len(valDataset)), shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    if saver is not None:
        context = logging_redirect_tqdm([saver.Logger])
        context.__enter__()
    trainer.train(prefetcher, trainSampler, valLoader, testLoader)
    if saver is not None:
        context.__exit__(None, None, None)

if __name__ == "__main__":
    app.run(main)
