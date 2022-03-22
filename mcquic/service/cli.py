import click

from .stCompressService import main


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-D", "--debug", is_flag=True, help="Set logging level to DEBUG to print verbose messages.")
@click.option("-q", "--quiet", is_flag=True, help="Silence all messages, this option has higher priority to `-D/--debug`.")
@click.option("-qp", type=click.IntRange(1, 13), default=2, show_default=True, help="Quantization parameter. Higher means better image quality and larger size.")
@click.option("-d", "--disable-gpu", is_flag=True, help="Disable GPU usage.")
def entryPoint(debug, quiet, qp, disable_gpu):
    """Start a Streamlit service for compressing and restoring images
    """
    raise NotImplementedError
