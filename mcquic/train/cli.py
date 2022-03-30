import logging
import pathlib
import random

import torch
import torch.multiprocessing as mp
import click
import yaml
from vlutils.runtime import queryGPU

from mcquic.config import Config
from .ddp import ddpSpawnTraining, registerForTrain

def checkArgs(debug, quiet, resume: pathlib.Path, configPath: pathlib.Path):
    if resume is None and configPath is None:
        raise ValueError("Both `resume` and `config` are not given.")
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO

def main(debug: bool, quiet: bool, resume: pathlib.Path, configPath: pathlib.Path) -> int:
    assert False, "You need to run `mcquic train` with Python optimized-mode. Try re-run me with `python -O -m mcquic.train ...`"
    loggingLevel = checkArgs(debug, quiet, resume, configPath)

    logging.getLogger().setLevel(loggingLevel)

    if configPath is not None:
        config = Config.deserialize(yaml.full_load(configPath.read_text()))
        logging.debug("Use fresh config.")
    elif resume is not None:
        ckpt = torch.load(resume, "cpu")
        config = Config.deserialize(ckpt["config"])
        logging.debug("Use saved config.")
    else:
        raise ValueError("Both `--resume` and `config` are None.")

    gpus = queryGPU(needGPUs=config.Train.GPU.GPUs, wantsMore=config.Train.GPU.WantsMore, needVRamEachGPU=(config.Train.GPU.VRam + 256) if config.Train.GPU.VRam > 0 else -1, writeOSEnv=True)
    worldSize = len(gpus)

    masterPort = str(random.randint(10001, 65535))

    registerForTrain()

    # `daemon` is True --- Way to handle SIGINT globally.
    # Give up handling SIGINT by yourself... PyTorch hacks it.
    mp.spawn(ddpSpawnTraining, (worldSize, masterPort, config, config.Train.SaveDir, resume, loggingLevel), worldSize, daemon=True)
    return 0


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-r", "--resume", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1, help="`.ckpt` file path to resume training.")
@click.argument('config', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def entryPoint(debug, quiet: bool, resume: pathlib.Path, config: pathlib.Path):
    main(debug, quiet, resume, config)
