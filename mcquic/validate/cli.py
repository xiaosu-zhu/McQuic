import os
import warnings
import click
import pathlib
import logging

import torch
from vlutils.logger import configLogging
from torchvision.io.image import write_png

import mcquic
from mcquic.config import Config
from mcquic.modules.compressor import Compressor
from mcquic.train.utils import getRichProgress
from mcquic.datasets import getValLoader
from mcquic.utils import hashOfFile, versionCheck

from .validator import Validator


def checkArgs(debug: bool, quiet: bool):
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO


def main(debug: bool, quiet: bool, export: pathlib.Path, path: pathlib.Path, images: pathlib.Path, output: pathlib.Path):
    loggingLevel = checkArgs(debug, quiet)

    logger = configLogging(None, "root", loggingLevel)

    checkpoint = torch.load(path, "cuda")

    config = Config.deserialize(checkpoint["config"])

    model = Compressor(**config.Model.Params).cuda().eval()

    if "trainer" in checkpoint:
        modelStateDict = {key[len("module._compressor."):]: value for key, value in checkpoint["trainer"]["_model"].items()}
    else:
        modelStateDict = checkpoint["model"]
        if export is not None:
            warnings.warn("I got an already-converted ckpt.")
        if not "version" in checkpoint:
            raise RuntimeError("You are using a too old version of ckpt, since there is no `version` in it.")
        versionCheck(checkpoint["version"])

    model.load_state_dict(modelStateDict)

    validator = Validator(config, "cuda")

    valLoader = getValLoader(images, False, logger)

    progress = getRichProgress()

    with progress:
        results, summary = validator.validate(None, model, valLoader, progress)
        logger.info(summary)
        _, speedSummary = validator.speed(None, model, progress)
        logger.info(speedSummary)

        if output is not None:
            allImages = results["ImageCollector"]

            total = len(allImages)

            task = progress.add_task(f"[ Save ]", total=total, progress=f"{0:4d}/{total:4d}", suffix="")

            for now, (image, stem) in enumerate(allImages):
                write_png(image, os.path.join(output, f"{stem}.png"))

                progress.update(task, advance=1, progress=f"{(now + 1):4d}/{total:4d}")
            progress.remove_task(task)

    if export is None:
        logger.info(f"Skip saving model.")
        return

    qp = config.Model.Params["m"]
    finalName = export.joinpath(f"qp_{qp}_{config.Train.Target.lower()}.mcquic")

    torch.save({
        "model": model.state_dict(),
        "config": config.serialize(),
        "version": mcquic.__version__
    }, finalName)

    logger.info(f"Saved at `{finalName}`.")
    logger.info("Add hash to file...")

    with progress:
        hashResult = hashOfFile(finalName, progress)

    newName = f"{finalName.stem}_{hashResult[:8]}{finalName.suffix}"

    os.rename(finalName, finalName.parent.joinpath(newName))

    logger.info("Rename file to %s", newName)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-e", "--export", type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, help="Dir to export the final model that is compatible with main program. Model name is generated automatically.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def entryPoint(debug, quiet, export, path, images, output):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    images (str): Validation images folder.

    output (str): Dir to save all restored images.
    """
    main(debug, quiet, export, path, images, output)
