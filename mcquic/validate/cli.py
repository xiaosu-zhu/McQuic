import os
import click
import pathlib
import logging

import mcquic


def checkArgs(debug: bool, quiet: bool, path: pathlib.Path, output: pathlib.Path):
    if path.is_dir():
        raise ValueError("Please provide a file path to `path`, not a dir.")
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO


def main(debug: bool, quiet: bool, path: pathlib.Path, images: pathlib.Path, output: pathlib.Path) -> int:
    loggingLevel = checkArgs(debug, quiet, path, output)

    import hashlib

    import torch
    from vlutils.logger import configLogging

    from mcquic.config import Config
    from mcquic.modules.compressor import Compressor
    from mcquic.train.utils import getRichProgress
    from mcquic.datasets import getValLoader

    from .validator import Validator

    logger = configLogging(None, "root", loggingLevel)

    checkpoint = torch.load(path, "cuda")

    config = Config.deserialize(checkpoint["config"])

    model = Compressor(**config.Model.Params).cuda().eval()

    if "trainer" in checkpoint:
        modelStateDict = {key[len("module._compressor."):]: value for key, value in checkpoint["trainer"]["_model"].items()}
    else:
        modelStateDict = checkpoint["model"]
        logger.warning("I got an already-converted ckpt. The `output` will be ignored. If you still want to export this ckpt, please copy it directly.")
        if not "version" in checkpoint or (checkpoint["version"] != mcquic.__version__):
            v = checkpoint.get("version", None)
            logger.warning(f"Version mismatch: It seems this ckpt has a version {v} but mcquic now is {mcquic.__version__}.")
        output = None

    model.load_state_dict(modelStateDict) # type: ignore

    validator = Validator(config, "cuda")

    valLoader = getValLoader(images, False, logger)

    progress = getRichProgress()

    with progress:
        _, summary = validator.validate(None, model, valLoader, progress)
        logger.info(summary)
        _, speedSummary = validator.speed(None, model, progress)
        logger.info(speedSummary)

    if output is None or (output.is_dir() and not output.exists()):
        logger.info(f"I got an invalid path: `{output}`, skip saving model.")
        return 0

    if output.is_dir():
        modelName = "_".join([f"{key}_{value}" for key, value in config.Model.Params.items()])
        modelName = modelName.replace(", ", "_").replace("[", "").replace("]", "")
        output = output.joinpath(f"{modelName}_{config.Train.Target.lower()}.mcquic")

    torch.save({
        "model": model.state_dict(),
        "config": config.serialize(),
        "version": mcquic.__version__
    }, output)

    logger.info(f"Saved at `{output}`.")
    logger.info("Add hash to file...")
    sha256 = hashlib.sha256()

    with open(output, 'rb') as fp:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = fp.read(65536)
            if not chunk:
                break
            sha256.update(chunk)

    hashResult = sha256.hexdigest()

    newName = f"{output.stem}-{hashResult[:8]}.{output.suffix}"

    os.rename(output, output.parent.joinpath(newName))

    logger.info("Rename file to %s", newName)

    return 0


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=False, dir_okay=True, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def entryPoint(debug, quiet, path, images, output):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    images (str): Validation images folder.

    output (str): File path or dir to publish this model.
    """
    main(debug, quiet, path, images, output)
