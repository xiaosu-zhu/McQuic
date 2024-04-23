import logging
import pathlib
import random
import os

import torch
import torch.multiprocessing as mp
import click
import yaml
from vlutils.runtime import queryGPU

from mcquic.config import Config
from mcquic.train.ddp import ddpSpawnTraining

def checkArgs(debug, quiet, configPath: pathlib.Path):
    if configPath is None:
        raise ValueError("`config` is not given.")
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO

def main(debug: bool, quiet: bool, gen: bool, configPath: pathlib.Path) -> int:
    # assert False, "You need to run `mcquic train` with Python optimized-mode. Try re-run me with `python -O -m mcquic.train ...`"
    loggingLevel = checkArgs(debug, quiet, configPath)

    logging.getLogger().setLevel(loggingLevel)

    config = Config.deserialize(yaml.full_load(configPath.read_text()))
    if os.path.exists(os.path.join(config.Train.SaveDir, 'latest', 'saved.ckpt')):
        resume = os.path.join(config.Train.SaveDir, 'latest', 'saved.ckpt')
        ckpt = torch.load(resume, "cpu")
        config = Config.deserialize(ckpt["config"])
        logging.info("Resume training from %s.", resume)
        resume = pathlib.Path(resume)
    else:
        resume = None
        logging.info("Start a fresh training. Checkpoint will be saved to %s.", os.path.join(config.Train.SaveDir, 'latest'))

    # gpus = queryGPU(needGPUs=config.Train.GPU.GPUs, wantsMore=config.Train.GPU.WantsMore, needVRamEachGPU=(config.Train.GPU.VRam + 256) if config.Train.GPU.VRam > 0 else -1, writeOSEnv=True)
    # worldSize = len(gpus)

    # masterPort = str(random.randint(10001, 65535))

    # `daemon` is True --- Way to handle SIGINT globally.
    # Give up handling SIGINT by yourself... PyTorch hacks it.
    ddpSpawnTraining(gen, config, config.Train.SaveDir, resume, loggingLevel)
    return 0


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-G", "--gen", is_flag=True, help="Enable stage-2 generation training.")
@click.argument('config', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def entryPoint(debug, quiet: bool, gen: bool, config: pathlib.Path):
    main(debug, quiet, gen, config)
