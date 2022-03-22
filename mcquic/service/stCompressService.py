import os
import pathlib
import torch
import torch.hub
from torchvision.transforms.functional import convert_image_dtype, pil_to_tensor
from torchvision.io.image import encode_png
from PIL import Image
import PIL

from mcquic import Config
from mcquic.modules.compressor import BaseCompressor, Compressor
from mcquic.datasets.transforms import AlignedCrop
from mcquic.utils.specification import File
from mcquic.utils.vision import DeTransform

try:
    import streamlit as st
except:
    raise ImportError("To run `mcquic service`, please install Streamlit by `pip install streamlit` firstly.")


MODELS_URL = "https://github.com/xiaosu-zhu/McQuic/releases/download/generic/qp_2_msssim_fcc58b73.mcquic"

HF_SPACE = "HF_SPACE" in os.environ


@st.experimental_singleton
def loadModel(qp: int, local: pathlib.Path, device, mse: bool):
    ckpt = torch.hub.load_state_dict_from_url(MODELS_URL, map_location=device, check_hash=True)

    config = Config.deserialize(ckpt["config"])
    model = Compressor(**config.Model.Params).to(device)
    model.QuantizationParameter = str(local) if local is not None else "qp_2_msssim"
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



def main(debug: bool, quiet: bool, qp: int, disable_gpu: bool):
    if disable_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = loadModel(3, None, device, False).eval()

    st.sidebar.markdown("""
<p align="center">
  <a href="https://github.com/xiaosu-zhu/McQuic" target="_blank">
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


<img src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>

> Due to resources limitation, I only provide compression service with model `qp = 2`.

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

<p align="center">
<a href="https://github.com/xiaosu-zhu/McQuic" target="_blank">
  <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/GitHub_Logo.png" height="16px" alt="Github"/>
  <img src="https://img.shields.io/github/stars/xiaosu-zhu/McQuic?style=social" height="20px" alt="Github"/>
</a>
</p>
""", unsafe_allow_html=True)

    if HF_SPACE:
        st.markdown("""
<img src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>

> Due to resources limitation of HF spaces, the upload image size is restricted to lower than `3000 x 3000`.

<img src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>

> Also, this demo running on HF space is GPU-disabled. So it may be slow.
""", unsafe_allow_html=True)
    with st.form("SubmitForm"):
        uploadedFile = st.file_uploader("Try running McQuic to compress or restore images!", type=["png", "jpg", "jpeg", "mcq"], help="Upload your image or compressed `.mcq` file here.")
        cropping = st.checkbox("Cropping image to align grids.", help="If checked, the image is cropped to align feature map grids. This will make compressed file smaller.")
        submitted = st.form_submit_button("Submit", help="Click to start compress/restore.")
    if submitted and uploadedFile is not None:
        if uploadedFile.name.endswith(".mcq"):
            uploadedFile.flush()

            binaryFile = File.deserialize(uploadedFile.read())

            st.text(str(binaryFile))

            result = decompressImage(binaryFile, model)
            st.image(result.cpu().permute(1, 2, 0).numpy())

            downloadButton = st.empty()

            done = downloadButton.download_button("Click to download restored image", data=bytes(encode_png(result.cpu()).tolist()), file_name=".".join(uploadedFile.name.split(".")[:-1] + ["png"]), mime="image/png")

            if done:
              downloadButton.empty()
        elif uploadedFile.name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(uploadedFile)
            except PIL.UnidentifiedImageError:
                st.markdown("""
<img src="https://img.shields.io/badge/ERROR-red?style=for-the-badge" alt="ERROR"/>

> Image open failed. Please try other images.
""", unsafe_allow_html=True)
                return
            w, h = image.size
            if HF_SPACE and (h > 3000 or w > 3000):
                st.markdown("""
<img src="https://img.shields.io/badge/ERROR-red?style=for-the-badge" alt="ERROR"/>

> Image is too large. Please try other images.
""", unsafe_allow_html=True)
                return
            image = pil_to_tensor(image.convert("RGB")).to(device)
            # st.image(image.cpu().permute(1, 2, 0).numpy())
            result = compressImage(image, model, cropping)

            st.text(str(result))

            downloadButton = st.empty()

            done = st.download_button("Click to download compressed file", data=result.serialize(), file_name=".".join(uploadedFile.name.split(".")[:-1] + ["mcq"]), mime="image/mcq")

            if done:
              downloadButton.empty()
        else:
            st.markdown("""
<img src="https://img.shields.io/badge/ERROR-red?style=for-the-badge" alt="ERROR"/>

> Not supported image formate. Please try other images.
""", unsafe_allow_html=True)
            return
    st.markdown("""
<br/>
<br/>
<br/>
<br/>
<br/>


<p align="center">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  </a>
  <a href="https://pytorch.org/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/stargazers" target="_blank">
    <img src="https://img.shields.io/github/stars/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github stars"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/network/members" target="_blank">
    <img src="https://img.shields.io/github/forks/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github forks"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github license"/>
  </a>
</p>

<br/>
<br/>
<br/>

<p align="center"><a href="localhost" target="_blank">CVF Open Access</a> | <a href="https://arxiv.org/abs/2203.10897" target="_blank">arXiv</a> | <a href="https://github.com/xiaosu-zhu/McQuic#citation" target="_blank">BibTex</a> | <a href="https://huggingface.co/spaces/xiaosu-zhu/McQuic" target="_blank">Demo</a></p>

""", unsafe_allow_html=True)

if __name__ == "__main__":
    with torch.inference_mode():
        main(False, False, 3, False)
