import pathlib
import click
from vlutils.utils import DefaultGroup


def version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('McQuic 0.0.9')
    ctx.exit()


@click.group(cls=DefaultGroup)
def entryPoint():
    pass


@entryPoint.command(default=True)
@click.option("-v", "--version", is_flag=True, callback=version, expose_value=False, is_eager=True, help="Print version info.")
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-qp", type=click.IntRange(1, 13), default=3, show_default=True, help="Quantization parameter. Higher means better image quality and larger size.")
@click.argument('input', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path), nargs=1)
@click.argument('output', type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path), required=False, nargs=1)
def _(debug, quiet, qp, input, output):
    """Compress/restore a file.

Args:

    input (str): Input file path. If input is an image, compress it. If input is a `.mcq` file, restore it.

    output (optional, str): Output file path or dir. If not provided, this program will only print compressor information of input file.
    """
    print(qp, input, output)


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
