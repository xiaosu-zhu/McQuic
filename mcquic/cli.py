import pathlib

import click
import torch
import torch.hub
from vlutils.utils import DefaultGroup

import mcquic



def version(ctx, _, value):
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
    from .demo import main
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
@click.option("-e", "--export", type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path), required=False, help="Path to export the final model that is compatible with main program.")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("images", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=True, nargs=1)
@click.argument("output", type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def validate(debug, quiet, export, path, images, output):
    """Validate a trained model from `path` by images from `images` dir, and publish a final state_dict to `output` path.

Args:

    path (str): Saved checkpoint path.

    images (str): Validation images folder.

    output (str): Dir to save all restored images.
    """
    from mcquic.validate.cli import main
    with torch.inference_mode():
        main(debug, quiet, export, path, images, output)


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
