
import os
from logging import Logger
import math
import random

import torch
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from absl import app
from absl import flags
from cfmUtils.runtime import queryGPU
from cfmUtils.logger import configLogging
from cfmUtils.saver import Saver
from cfmUtils.config import read, summary
from cfmUtils.vision.utils import verifyTruncated

from mcqc import Consts, Config
from mcqc.datasets import Basic
from mcqc.algorithms import Plain
from mcqc.models.whole import Whole, WholeVQ, WholeRein, WholeTwoStage, WholeTwoStageWithGan, WholePQMLM
from mcqc.models.discriminator import Discriminator, FullDiscriminator
from mcqc.utils import getTrainingTransform, getEvalTransform

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
    config.lr *= math.sqrt(batchSize)

def _generalConfig(rank: int, worldSize: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29811"
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(rank)
    random.seed(rank)
    np.random.seed(rank)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dist.init_process_group("nccl", world_size=worldSize, rank=rank)
    dist.barrier()

# def Test(config: Config, saveDir: str, logger: Logger = None) -> None:
#     _ = queryGPU(needGPUs=1, wantsMore=False, needVRamEachGPU=18000)
#     dataset = SiftLike(config.Dataset).Train()
#     _, D = dataset.shape
#     paramsForEnv = {
#         "m": config.HParams.M,
#         "k": config.HParams.K,
#         "d": D,
#         "doNormalizeOnObs": config.HParams.NormalizeObs,
#         "doNormalizeOnRew": config.HParams.NormalizeRew
#     }
#     config.HParams.__dict__.update({'d': D})
#     paramsForActorCritic = config.HParams.__dict__
#     methods = {
#         "PPO": ActorCritic,
#         "SAC": GumbelActorCritic,
#         "A2C": None,
#         "INC": InceptAC,
#         "NOVC": ActorCritic
#     }

#     ConfigLogging(saveDir, Consts.LoggerName, "DEBUG" if FLAGS.debug else "INFO", rotateLogs=-1, logName="eval")

#     (logger or Consts.Logger).info(str(config))
#     runner = Eval(False, os.path.join(saveDir, Consts.CheckpointName), dataset, Env(**paramsForEnv), methods[config.Method](**paramsForActorCritic))
#     runner.Test()

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
    model = WholePQMLM(config.Model.k, config.Model.channel, config.Model.numLayers)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def optimWrapper(lr, params, weight_decay):
        return torch.optim.AdamW(params, lr, amsgrad=True, eps=Consts.Eps, weight_decay=weight_decay)
    def schdrWrapper(optim):
        return torch.optim.lr_scheduler.ExponentialLR(optim, 0.5)
    method = Plain(config, model, optimWrapper, schdrWrapper, saver, savePath, continueTrain, logger)

    trainDataset = Basic(os.path.join("data", config.Dataset), transform=getTrainingTransform())

    trainSampler = torch.utils.data.DistributedSampler(trainDataset, worldSize, rank)
    # batchSampler = torch.utils.data.BatchSampler(trainSampler, config.BatchSize, drop_last=True)

    trainLoader = torch.utils.data.DataLoader(trainDataset, sampler=trainSampler, batch_size=config.BatchSize, num_workers=config.BatchSize + 4, pin_memory=True, drop_last=False)
    valLoader = torch.utils.data.DataLoader(Basic(os.path.join("data", config.ValDataset), transform=getEvalTransform()), batch_size=config.BatchSize * 4, shuffle=False, num_workers=worldSize * 4, pin_memory=True, drop_last=False)
    method.run(trainLoader, trainSampler, valLoader if rank == 0 else None)


if __name__ == "__main__":
    app.run(main)
