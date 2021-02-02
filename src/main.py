
import os
from logging import Logger

import torch
import torchvision

from absl import app
from absl import flags
from cfmUtils.runtime import queryGPU
from cfmUtils.logger import configLogging
from cfmUtils.saver import Saver
from cfmUtils.config import read, summary
from cfmUtils.vision.utils import verifyTruncated

from mcqc import Consts, Config
from mcqc.datasets import Basic
from mcqc.algorithms import Plain, PlainWithGAN
from mcqc.models.compressor import Compressor, MultiScaleCompressor
from mcqc.models.discriminator import Discriminator
from mcqc.utils import getTrainingTransform, getEvalTransform

torch.backends.cudnn.benchmark = True
FLAGS = flags.FLAGS

flags.DEFINE_string("config", "", "The config.json path.")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("continue", False, "Be careful to set to true. Whether to continue last training (with current config).")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.")


def main(_):
    if FLAGS.eval:
        assert FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace(), f"When --eval, --path must be set, got {FLAGS.path}."
        os.makedirs(FLAGS.path, exist_ok=True)
        saveDir = FLAGS.path
        config = read(os.path.join(saveDir, Consts.DumpConfigName), None, Config)
        # Test(config, saveDir)
    else:
        config = read(FLAGS.config, None, Config)
        if FLAGS.path is not None and len(FLAGS.path) > 0 and not FLAGS.path.isspace():
            os.makedirs(FLAGS.path, exist_ok=True)
            saveDir = FLAGS.path
        else:
            saveDir = os.path.join(Consts.SaveDir, config.Dataset)
        Train(config, saveDir)

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

def Train(config: Config, saveDir: str, logger: Logger = None) -> None:
    gpus = queryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.VRam + 256) if config.VRam > 0 else -1)

    saver = Saver(saveDir, "saved.ckpt", config, reserve=FLAGS.get_flag_value("continue", False))

    logger = configLogging(saver.SaveDir, Consts.LoggerName, "DEBUG" if FLAGS.debug else "INFO", rotateLogs=-1)

    logger.info("\r\n%s", summary(config))

    model = MultiScaleCompressor()
    dis = Discriminator(512)
    method = PlainWithGAN(model, dis, "cuda", lambda lr, params, weight_decay: torch.optim.Adam(params, lr, amsgrad=True, eps=Consts.Eps, weight_decay=weight_decay), lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, 0.8), saver, FLAGS.get_flag_value("continue", False), logger, config.Epoch)

    method.run(torch.utils.data.DataLoader(Basic(os.path.join("data", config.Dataset), transform=getTrainingTransform()), batch_size=config.BatchSize, shuffle=True, num_workers=len(gpus) * 4, pin_memory=True), torch.utils.data.DataLoader(Basic(os.path.join("data", config.ValDataset), transform=getEvalTransform()), batch_size=config.BatchSize, shuffle=True, num_workers=len(gpus) * 4, pin_memory=True, drop_last=True))


if __name__ == "__main__":
    app.run(main)
