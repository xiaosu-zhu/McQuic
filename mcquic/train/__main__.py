import logging
import os
import random

import torch.multiprocessing as mp
from absl import app
from absl import flags
from vlutils.runtime import queryGPU
from vlutils.config import read

from mcquic import Consts, Config
from . import train

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "", "The `config.yaml` path.", short_name="c")
flags.DEFINE_string("resume", None, "If resume is not None, load checkpoint from path `resume` and resume training", short_name="r")
flags.DEFINE_boolean("debug", False, "Set to true to log verbosely.", short_name="D")


def main(_):
    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = read(FLAGS.config, None, Config)
    saveDir = os.path.join(Consts.SaveDir, config.Dataset)
    gpus = queryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.VRam + 256) if config.VRam > 0 else -1, writeOSEnv=True)
    worldSize = len(gpus)
    config.scaleByWorldSize(worldSize)
    masterPort = str(random.randint(10001, 65535))
    # `daemon` is True --- Way to handle SIGINT globally.
    # Give up handling SIGINT by yourself... PyTorch hacks it.
    mp.spawn(train, (worldSize, masterPort, config, saveDir, FLAGS.resume, FLAGS.debug), worldSize, daemon=True) # type: ignore

if __name__ == "__main__":
    app.run(main)
