import logging
import pathlib
from typing import Optional, Tuple, Union
import warnings

import torch
import torch.hub
from torchvision.io.image import read_image, ImageReadMode, write_png
from torchvision.transforms.functional import convert_image_dtype
from vlutils.logger import configLogging

from mcquic import Config
from mcquic.modules.compressor import BaseCompressor, Compressor
from mcquic.utils import versionCheck
from mcquic.utils.specification import File
from mcquic.utils.vision import DeTransform


MODELS_URL = "https://github.com/xiaosu-zhu/McQuic/releases/download/generic/"

MODELS_HASH = {
    "qp_2_msssim": "8e954998"
}


def checkArgs(debug: bool, quiet: bool):
    if quiet:
        return logging.CRITICAL
    if debug:
        return logging.DEBUG
    return logging.INFO


def main(debug: bool, quiet: bool, qp: int, local: pathlib.Path, disable_gpu: bool, mse: bool, crop: bool, input: pathlib.Path, output: pathlib.Path):
    loggingLevel = checkArgs(debug, quiet)

    if disable_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    logger = configLogging(None, "root", loggingLevel)

    if input.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        model = loadModel(qp, local, device, mse, logger)

        image = read_image(str(input), ImageReadMode.RGB).to(device)

        target = compressImage(image, model, crop)

        logger.info(target)
        if output is not None:
            if output.is_dir():
                output = output.joinpath(input.stem + ".mcq")
            with open(output, "wb") as fp:
                fp.write(target.serialize())
            logger.info("Saved at %s", output)

    elif input.suffix.lower() == ".mcq":
        with open(input, "rb") as fp:
            binary = fp.read()
            source = File.deserialize(binary)

            model = detectModelFromFile(qp, local, mse, device, logger, source)

            restored = decompressImage(source, model)

            logger.info(source)

            if output is not None:
                if output.is_dir():
                    output = output.joinpath(input.stem + ".png")
                write_png(restored.cpu(), str(output))
    else:
        raise ValueError("Invalid input file.")

def detectModelFromFile(qp, local, mse, device, logger, source):
    localFile = detectLocalFile(source.FileHeader.QuantizationParameter)
    if isinstance(localFile, pathlib.Path):
        model = loadModel(-1, localFile, device, False, logger)
    else:
        parsed = parseQP(source.FileHeader.QuantizationParameter)
        if parsed is not None:
            model = loadModel(parsed[0], None, device, parsed[1], logger)
        else:
            warnings.warn("All qp detections failed. Fallback to use current args or you could try again after checks.")
            model = loadModel(qp, local, device, mse, logger)
    return model


def detectLocalFile(qp: str) -> Union[bool, pathlib.Path]:
    filePath = pathlib.Path(qp)
    if filePath.exists() and filePath.is_file() and "mcquic" in filePath.suffix.lower():
        return filePath
    return False

def parseQP(qp: str) -> Union[None, Tuple[int, bool]]:
    try:
        if not qp.startswith("qp_"):
            return None
        # qp_x_[mse/msssim]
        parsed = qp.split("_")
        return int(parsed[1]), parsed[2] == "mse"
    except:
        return None

def compressImage(image: torch.Tensor, model: BaseCompressor, crop: bool) -> File:
    image = convert_image_dtype(image)

    if crop:
        from mcquic.datasets.transforms import AlignedCrop
        image = AlignedCrop()(image)

    # [c, h, w]
    image = (image - 0.5) * 2

    with model.readyForCoding() as cdfs:
        _, binaries, headers = model.compress(image[None, ...], cdfs)

    # List of each level binary, FileHeader
    return File(headers[0], binaries[0])


def decompressImage(sourceFile: File, model: BaseCompressor) -> torch.Tensor:

    binaries = sourceFile.Content

    with model.readyForCoding() as cdfs:
        # append it to list to make batch-size = 1.
        # [1, c, h, w]
        restored = model.decompress([binaries], cdfs, [sourceFile.FileHeader])

    # [c, h, w]
    return DeTransform()(restored[0])


def loadModel(qp: int, local: Optional[pathlib.Path], device, mse: bool, logger: logging.Logger) -> BaseCompressor:
    # Fallback key
    key = "qp_2_msssim"
    if local is not None:
        warnings.warn(f"By passing `--local`, `-qp` arg will be ignored. Checkpoint from {local} will be loaded. Please ensure you obtain this local model from a trusted source.")
        ckpt = torch.load(local, device)

        logger.info("Use local model.")
    else:
        suffix = "mse" if mse else "msssim"
        key = f"qp_{qp}_{suffix}"
        if key not in MODELS_HASH:
            raise ValueError(f"The provided {key} combination not found in pretrained models.")
        ckpt = torch.hub.load_state_dict_from_url(MODELS_URL + f"qp_{qp}_{suffix}_{MODELS_HASH[key]}.mcquic", map_location=device, check_hash=True)

        logger.info("Use model `-qp %d` targeted `%s`.", qp, suffix)

    if not "version" in ckpt:
        raise RuntimeError("You are using a too old ckpt where `version` not in it.")
    versionCheck(ckpt["version"])

    config = Config.deserialize(ckpt["config"])
    model = Compressor(**config.Model.Params).to(device).eval()
    model.QuantizationParameter = str(local) if local is not None else key
    model.load_state_dict(ckpt["model"])
    logger.info(f"Model loaded, params: {config.Model.Params}.")
    return model
