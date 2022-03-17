import pathlib
import torch
import torch.hub
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io.image import ImageReadMode, encode_png, decode_image

from mcquic import Config
from mcquic.modules.compressor import BaseCompressor, Compressor
from mcquic.datasets.transforms import AlignedCrop
from mcquic.utils.specification import File
from mcquic.utils.vision import DeTransform

try:
    import streamlit as st
except:
    raise ImportError("To run `mcquic service`, please install Streamlit by `pip install streamlit` firstly.")


MODELS_URL = "https://github.com/xiaosu-zhu/McQuic/releases/download/generic/qp_3_msssim_fcc58b73.mcquic"


@st.experimental_singleton
def loadModel(qp: int, local: pathlib.Path, device, mse: bool):
    ckpt = torch.hub.load_state_dict_from_url(MODELS_URL, map_location=device, check_hash=True)

    config = Config.deserialize(ckpt["config"])
    model = Compressor(**config.Model.Params).to(device)
    model.QuantizationParameter = str(local) if local is not None else str(qp)
    model.load_state_dict(ckpt["model"])
    return model



@st.cache
def compressImage(image: torch.Tensor, model: BaseCompressor, crop: bool) -> File:
    image = convert_image_dtype(image)

    if crop:
        image = AlignedCrop()(image)

    # [c, h, w]
    image = (image - 0.5) * 2

    with model._quantizer.readyForCoding() as cdfs:
        codes, binaries, headers = model.compress(image[None, ...], cdfs)

    return File(headers[0], binaries[0])


@st.cache
def decompressImage(sourceFile: File, model: BaseCompressor) -> torch.ByteTensor:
    binaries = sourceFile.Content

    with model._quantizer.readyForCoding() as cdfs:
        # [1, c, h, w]
        restored = model.decompress([binaries], cdfs, [sourceFile.FileHeader])

    # [c, h, w]
    return DeTransform()(restored[0])



def main(debug: bool, quiet: bool, disable_gpu: bool):
    if disable_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = loadModel(3, None, device, False).eval()

    st.sidebar.markdown("""
<p align="center">
  <a href="https://github.com/xiaosu-zhu/McQuic">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/McQuic-light.svg" alt="McQuic" title="McQuic" width="45%"/>
  </a>
  <br/>
  <span>
    <i>a.k.a.</i> <b><i>M</i></b>ulti-<b><i>c</i></b>odebook <b><i>Qu</i></b>antizers for neural <b><i>i</i></b>mage <b><i>c</i></b>ompression
  </span>
</p>

<p align="center">
  Compressing images on-the-fly.
</p>


<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> Due to resources limitation, I only provide compression service with model `qp = 3`.


<a href="#">
  <image src="https://img.shields.io/github/stars/xiaosu-zhu/McQuic?style=social" alt="Github"/>
</a>

""", unsafe_allow_html=True)


    with st.form("SubmitForm"):
        uploadedFile = st.file_uploader("Try running McQuic to compress or restore images!", type=["png", "jpg", "jpeg", "mcq"], help="Upload your image or compressed `.mcq` file here.")
        cropping = st.checkbox("Cropping image to align grids.", help="If checked, the image is cropped to align to feature map grids. This makes output smaller.")
        submitted = st.form_submit_button("Submit", help="Click to start compress/restore.")
    if submitted and uploadedFile is not None:
        if uploadedFile.name.endswith(".mcq"):
            uploadedFile.flush()

            binaryFile = File.deserialize(uploadedFile.read())

            st.text(str(binaryFile))

            result = decompressImage(binaryFile, model)
            st.image(result.cpu().permute(1, 2, 0).numpy())
            st.download_button("Click to download restored image", data=bytes(encode_png(result.cpu()).tolist()), file_name=".".join(uploadedFile.name.split(".")[:-1] + ["png"]), mime="image/png")
        else:
            raw = torch.ByteTensor(torch.ByteStorage.from_buffer(uploadedFile.read())) # type: ignore
            image = decode_image(raw, ImageReadMode.RGB).to(device)
            st.image(image.cpu().permute(1, 2, 0).numpy())
            result = compressImage(image, model, cropping)

            st.text(str(result))

            st.download_button("Click to download compressed file", data=result.serialize(), file_name=".".join(uploadedFile.name.split(".")[:-1] + ["mcq"]), mime="image/mcq")


if __name__ == "__main__":
    with torch.inference_mode():
        main(False, False, False)
