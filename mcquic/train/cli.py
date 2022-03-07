import logging
import pathlib
import random


def main(debug: bool, quiet: bool, resume: pathlib.Path, configPath: pathlib.Path) -> int:
    from .ddp import ddpSpawnTraining, registerForTrain
    import torch.multiprocessing as mp
    from vlutils.runtime import queryGPU
    from mcquic.config import Config
    import yaml

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config.deserialize(yaml.full_load(configPath.read_text()))

    gpus = queryGPU(needGPUs=config.GPU.GPUs, wantsMore=config.GPU.WantsMore, needVRamEachGPU=(config.GPU.VRam + 256) if config.GPU.VRam > 0 else -1, writeOSEnv=True)
    worldSize = len(gpus)
    config.scaleByWorldSize(worldSize)

    masterPort = str(random.randint(10001, 65535))

    registerForTrain()

    # `daemon` is True --- Way to handle SIGINT globally.
    # Give up handling SIGINT by yourself... PyTorch hacks it.
    mp.spawn(ddpSpawnTraining, (worldSize, masterPort, config, config.Training.SaveDir, resume, debug), worldSize, daemon=True) # type: ignore
    return 0

def entryPoint():
    from absl import app
    app.run(main)
