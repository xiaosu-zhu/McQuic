import click

from .stCompressService import main


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-d", "--disable-gpu", is_flag=True, help="Disable GPU usage.")
def entryPoint(debug, quiet, disable_gpu):
    """Create training set from `images` dir to `output` dir.

Args:

    images (str): All training images folder, allow sub-folders.

    output (str): Output dir to create training set.
    """
    main(debug, quiet, disable_gpu)
