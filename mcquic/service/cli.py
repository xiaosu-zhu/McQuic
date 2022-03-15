import pathlib
import click
import torch
import torch.hub
from torchvision.transforms.functional import convert_image_dtype, to_tensor
from torchvision.io.image import read_image, ImageReadMode, write_png

from mcquic import Config
from mcquic.modules.compressor import Compressor
import mcquic
from mcquic.utils.specification import File
from mcquic.utils.vision import DeTransform

try:
    import gradio as gr
    import gradio.inputs as gr_inputs
    import gradio.outputs as gr_outputs
except:
    raise ImportError("To run `mcquic service`, please install Gradio by `pip install gradio` firstly.")


MODELS_URL = "https://github.com/xiaosu-zhu/McQuic/releases/download/generic/"

def loadModel(qp: int, local: pathlib.Path, device, mse: bool):

    suffix = "mse" if mse else "msssim"
    ckpt = torch.hub.load_state_dict_from_url(MODELS_URL + f"qp_{qp}_{suffix}.mcquic", map_location=device)

    config = Config.deserialize(ckpt["config"])
    model = Compressor(**config.Model.Params).to(device)
    model.QuantizationParameter = str(local) if local is not None else str(qp)
    model.load_state_dict(ckpt["model"])
    return model


def main(debug: bool, quiet: bool, disable_gpu: bool):
    if disable_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # model = loadModel(3, None, device, False).eval()

    def servingLoop(file):
        print(type(file))
        print(file)
        print(file.name)


    gr.Interface(fn=servingLoop, inputs=gr_inputs.File(), outputs=gr_outputs.File(), examples=["assets/sample.png"], title="McQuic", description="a.k.a. Multi-Codebook QUantizers for Image Compression", theme="huggingface").launch(show_tips=True, enable_queue=True, cache_examples=True)

def compressImage(image, model, crop):

    image = convert_image_dtype(image)

    if crop:
        from mcquic.datasets.transforms import AlignedCrop
        image = AlignedCrop()(image)

    # [c, h, w]
    image = (image - 0.5) * 2

    with model._quantizer.readyForCoding() as cdfs:
        codes, binaries, headers = model.compress(image[None, ...], cdfs)

    # List of each level binary, FileHeader
    return binaries[0], headers[0]


def decompressImage(sourceFile, model):
    binaries = sourceFile.Content

    with model._quantizer.readyForCoding() as cdfs:
        # [1, c, h, w]
        restored = model.decompress([binaries], cdfs, [sourceFile.FileHeader])

    # [c, h, w]
    return DeTransform()(restored[0])


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
