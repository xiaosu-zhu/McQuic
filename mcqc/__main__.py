import logging
import os

import torch.multiprocessing as mp
from absl import app
from absl import flags
from vlutils.runtime import queryGPU
from vlutils.config import read

from mcqc import Consts, Config
from mcqc.training import train

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "", "The config.yaml path.", short_name="c")
flags.DEFINE_string("path", "", "Specify saving path, otherwise use default pattern. In eval mode, you must specify this path where saved checkpoint exists.")
flags.DEFINE_string("master_port", "29936", "MASTER_PORT of DistributedDataParallel.")
flags.DEFINE_boolean("eval", False, "Evaluate performance. Must specify arg 'path', and arg 'config' will be ignored.")
flags.DEFINE_boolean("resume", False, "Be careful to set to true. Whether to continue last training (with current config).", short_name="r")
flags.DEFINE_boolean("debug", False, "Set to true to logging verbosely and require lower gpu.", short_name="D")


def main(_):
    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if FLAGS.eval:
        if FLAGS.path is None or (len(FLAGS.path) > 0 and FLAGS.path.isspace()):
            raise ValueError(f"When --eval, --path must be set, got {FLAGS.path}.")
        if not os.path.exists(FLAGS.path):
            raise ValueError(f"Invalid --path={FLAGS.path}, no such directory.")
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
        gpus = queryGPU(needGPUs=config.GPUs, wantsMore=config.WantsMore, needVRamEachGPU=(config.VRam + 256) if config.VRam > 0 else -1, writeOSEnv=True)
        worldSize = len(gpus)
        config.scaleByWorldSize(worldSize)
        # `daemon` is True --- Way to handle SIGINT globally.
        mp.spawn(train, (worldSize, FLAGS.master_port, config, saveDir, FLAGS.resume, FLAGS.debug), worldSize, daemon=True) # type: ignore

if __name__ == "__main__":
    app.run(main)
