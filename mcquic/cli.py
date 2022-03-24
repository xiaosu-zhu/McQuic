import logging
import pathlib

import click
import torch
import torch.hub
from torchvision.io.image import read_image, ImageReadMode, write_png
from torchvision.transforms.functional import convert_image_dtype
from vlutils.utils import DefaultGroup
from vlutils.logger import configLogging

import mcquic
from mcquic import Config
from mcquic.validate.cli import main
from mcquic.modules.compressor import Compressor
from mcquic.utils import versionCheck
from mcquic.utils.specification import File

MODELS_URL = "https://github.com/xiaosu-zhu/McQuic/releases/download/generic/"

MODELS_HASH = {
    "qp_2_msssim": "fcc58b73"
}


def version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(r"""
                                       :::::.:::.
#####*       +#####                  ::         .=   .:...+=   -...== -...-+    ........=-
%%%%%%+     -%%%%%%                 -     -=.    -#  :    %*  :    ##.:   =%  .:        #+
%%%%%%%:   .#%%%%%%     :=+***+-   -    -#*-+    .%: :   -%:  -   :%=-    #+ :.   .=+++*%:
%%%%#%%#.  #%%#%%%%  .+%%%%%%%%*  -    -%=  =    :%-:    ##  ::   *%.-   -%.-    =%+:
%%%%=%%%* +%%%=%%%% .#%%%%+::--   -   .%*...+    +%.:   -%=  =   .%=:    #*.:   -%-
%%%%-=%%%=%%%-=%%%% +%%%%*       :.   -*    .   .%*..   *%  :.   +% -   -%--    #+
%%%%= *%%%%%* +%%%% +%%%%+       -    :%=       *# :    #* .:   :%=::   *# -    #-
%%%%=  #%%%#  *%%%% :%%%%%-  .-  .:    -=      *%: :    -+:.   .#* =   .%= -    -=::::
%%%%=  :%%%.  *%%%%  -#%%%%%%%%*  -           .#:   -         -%* .:   :%. .:        *+
****=         =****    -+*#%##*=.  :::...:-:   .+    :-::.:-+%*-  -   .+%   .:-:....-%:
                                     .-====--====*.    :-===:     -.-*%+:      :======
                                             :::::.              :##+:
""" + mcquic.__version__)
    ctx.exit()


def checkArgs(debug, quiet):
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

        binaries, header = compressImage(image, model, crop)

        target = File(header, binaries)

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

            newLocal = None
            newQP = -1
            newMSE = False

            try:
                # qp_x_[mse/msssim]
                parsed = source.FileHeader.QuantizationParameter.split("_")
                newQP = int(parsed[1])
                newMSE = parsed[2] == "mse"
                newLocal = None
            except:
                newLocal = pathlib.Path(source.FileHeader.QuantizationParameter)
                newQP = -1
            finally:
                if newLocal is None and newQP < 0:
                    logger.warning("The compressed binary is produced by a pre-release model since `qp` in file is -1, fallback to use current args.")
                elif isinstance(newLocal, pathlib.Path) and not(newLocal.exists() and newLocal.is_file()):
                    logger.warning("The compressed binary is compressed by a local model located in `%s`. Unfortunately, we can't find it. Fallback to use current args or you could try again later.", newLocal)
                else:
                    qp = newQP
                    local = newLocal
                    mse = newMSE

            model = loadModel(qp, local, device, mse, logger).eval()

            restored = decompressImage(source, model)

            logger.info(source)

            if output is not None:
                if output.is_dir():
                    output = output.joinpath(input.stem + ".png")
                write_png(restored.cpu(), str(output))
    else:
        raise ValueError("Invalid input file.")

def compressImage(image, model, crop):
    image = convert_image_dtype(image)

    if crop:
        from mcquic.datasets.transforms import AlignedCrop
        image = AlignedCrop()(image)

    # [c, h, w]
    image = (image - 0.5) * 2

    with model._quantizer.readyForCoding() as cdfs:
        _, binaries, headers = model.compress(image[None, ...], cdfs)

    # List of each level binary, FileHeader
    return binaries[0], headers[0]


def decompressImage(sourceFile, model):
    binaries = sourceFile.Content

    with model._quantizer.readyForCoding() as cdfs:
        # [1, c, h, w]
        restored = model.decompress([binaries], cdfs, [sourceFile.FileHeader])

    from mcquic.utils.vision import DeTransform

    # [c, h, w]
    return DeTransform()(restored[0])


def loadModel(qp: int, local: pathlib.Path, device, mse: bool, logger: logging.Logger):
    if local is not None:
        logger.warning("By passing `--local`, `-qp` arg will be ignored and model from %s will be loaded. Please ensure you obtain this local model from a trusted source.", local)
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
    model = Compressor(**config.Model.Params).to(device)
    model.QuantizationParameter = str(local) if local is not None else key
    model.load_state_dict(ckpt["model"])
    logger.info(f"Model loaded, params: {config.Model.Params}.")
    return model


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(cls=DefaultGroup, context_settings=CONTEXT_SETTINGS)
@click.option("-v", "--version", is_flag=True, callback=version, expose_value=False, is_eager=True, help="Print version info.")
def entryPoint():
    pass


@entryPoint.command(default=True)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-qp", type=click.IntRange(0, 13), default=2, show_default=True, help="Quantization parameter. Higher means better image quality and larger size.")
@click.option("--local", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), help="Use a local model path instead of download by `qp`.")
@click.option("--disable-gpu", is_flag=True, help="Use pure CPU to perform compression. This will be slow.")
@click.option("--mse", is_flag=True, help="Use model optimized for PSNR other than MsSSIM.")
@click.option("--crop", is_flag=True, help="Crop the image to align feature patches. Edges of image are cutted though, compressed binary will be smaller.")
@click.argument('input', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), nargs=1)
@click.argument('output', type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def _(debug, quiet, qp, local, disable_gpu, mse, crop, input, output):
    """Compress/restore a file.

Args:

    input (str): Input file path. If input is an image, compress it. If input is a `.mcq` file, restore it.

    output (optional, str): Output file path or dir. If not provided, this program will only print compressor information of input file.
    """
    import torch
    with torch.inference_mode():
        main(debug, quiet, qp, local, disable_gpu, mse, crop, input, output)


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-r", "--resume", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1, help="`.ckpt` file path to resume training.")
@click.argument('config', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def train(debug, quiet, resume, config):
    """Train a model.

Args:

    config (str): Config file (yaml) path. If `-r/--resume` is present but config is still given, then this config will be used to update the resumed training.
    """
    from mcquic.train.cli import main
    main(debug, quiet, resume, config)


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=False, dir_okay=True, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def validate(debug, quiet, path, images, output):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    images (str): Validation images folder.

    output (str): File path or dir to publish this model.
    """
    with torch.inference_mode():
        main(debug, quiet, path, images, output)


@entryPoint.command()
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
def dataset(debug, quiet, images, output):
    """Create training set from `images` dir to `output` dir.

Args:

    images (str): All training images folder, allow sub-folders.

    output (str): Output dir to create training set.
    """
    from mcquic.datasets.cli import main
    main(images, output)
