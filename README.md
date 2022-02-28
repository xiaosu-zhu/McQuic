<p align="center">
  <a href="https://github.com/xiaosu-zhu/McQuic#gh-light-mode-only">
    <img src="./assets/McQuic-light.svg#gh-light-mode-only" alt="McQuic" title="McQuic" width="33%">
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic#gh-dark-mode-only">
    <img src="./assets/McQuic-dark.svg#gh-dark-mode-only" alt="McQuic" title="McQuic" width="33%">
  </a>
  <br/>
  <span>
    <i>a.k.a.</i> <b><i>M</i></b>ulti-<b><i>c</i></b>odebook <b><i>Qu</i></b>antizers for neural <b><i>i</i></b>mage <b><i>c</i></b>ompression
  </span>
</p>


<p align="center">
  <a href="https://www.python.org/">
    <image src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  </a>
  <a href="https://pytorch.org/">
    <image src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/commits/main">
    <image src="https://img.shields.io/github/commit-activity/m/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github commits"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/stargazers">
    <image src="https://img.shields.io/github/stars/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github stars"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/network/members">
    <image src="https://img.shields.io/github/forks/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github forks"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic/blob/main/LICENSE">
    <image src="https://img.shields.io/github/license/xiaosu-zhu/McQuic?logo=github&style=for-the-badge" alt="Github license"/>
  </a>
</p>


Multi-codebook quantizers hold rich codebooks to quantize visual features and restore images by these quantized features. Similar ideas represent in SHA [[1](#reference-and-license)], VQ-VAE [[2](#reference-and-license)], VQ-GAN [[3](#reference-and-license)], *etc*. We summarize these as vectorized priors, and our method extends these ideas to a ***unified multivariate Gaussian mixture***, to perform high-quality, low-latency image compression.

Take a more look at ***our paper***:

**Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression**



**[CVPR Open Access](localhost) | [arXiv](localhost) | [BibTex](localhost) | Demo will be online soon**

<p align="center">
    <img src="./assets/paper/priors-light.svg#gh-light-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <img src="./assets/paper/priors-dark.svg#gh-dark-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <span><b>Figure 1. Operational diagrams of different methods.</b></span>
</p>


<p align="center">
    <img src="./assets/paper/kodim24-light.png#gh-light-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <img src="./assets/paper/kodim24-dark.png#gh-dark-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <span><b>Figure 2. Comparisons with traditional codecs on an image from Kodak dataset.</b></span>
</p>

<!--ts-->
* [Introduction](#introduction)
* [Try Me!](#try-me)
   * [Requirements](#requirements)
   * [Docker (Recommended)](#docker-recommended)
   * [Install via PyPI](#install-via-pypi)
   * [Install Manually (for dev)](#install-manually-for-dev)
* [Develop, Contribute, or Train a New Model](#develop-contribute-or-train-a-new-model)
   * [Requirements](#requirements-1)
   * [Configs](#configs)
   * [Train and Test](#train-and-test)
* [To-do List](#to-do-list)
* [Reference and License](#reference-and-license)

<!-- Added by: runner, at: Sun Feb 27 15:08:52 UTC 2022 -->

<!--te-->



# Introduction
Thanks for your attention!❤️ Next we would like to tell some details...

Following previous works, we build the compression model as an AutoEncoder. Bottleneck of encoder (analysis transform) outputs a small feature map and is quantized by *multi-codebook vector-quantization* other than scalar-quantization. Quantizers are cascaded to effectively estimate latent distribution.

<p align="center">
    <img src="./assets/paper/framework-light.svg#gh-light-mode-only" alt="Framework" title="Framework" width="100%">
    <img src="./assets/paper/framework-dark.svg#gh-dark-mode-only" alt="Framework" title="Framework" width="100%">
    <span><b>Figure 3. Left: Overall framework. Right: Structure of a quantizer.</b></span>
</p>

Right part of above figure shows detailed structure of our proposed quantizer. A minimal implementation comes up with:

```python
class Quantizer(nn.Module):
    """
    Quantizer with `m` sub-codebooks,
        `k` codewords for each, and
        `n` total channels.
    Args:
        m (int): Number of sub-codebooks.
        k (int): Number of codewords for each sub-codebook.
        n (int): Number of channels of latent variables.
    """
    def __init__(self, m: int, k: int, n: int):
        super().__init__()
        # A codebook, channel `n -> n // m`.
        self._codebook = nn.Parameter(torch.empty(m, k, n // m))
        self._initParameters()

    def forward(self, x: Tensor, t: float = 1.0) -> (Tensor, Tensor):
        """
        Module forward.
        Args:
            x (Tensor): Latent variable with shape [b, n, h, w].
            t (float, 1.0): Temperature for Gumbel softmax.
        Return:
            Tensor: Quantized latent with shape [b, n, h, w].
            Tensor: Binary codes with shape [b, m, h, w].
        """
        b, _, h, w = x.shape
        # [b, m, d, h, w]
        x = x.reshape(n, len(self._codebook), -1, h, w)
        # [b, m, 1, h, w], square of x
        x2 = (x ** 2).sum(2, keepdim=True)
        # [m, k, 1, 1], square of codebook
        c2 = (self._codebook ** 2).sum(-1, keepdim=True)[..., None]
        # [b, m, d, h, w] * [m, k, d] -sum-> [b, m, k, h, w], dot product between x and codebook
        inter = torch.einsum("bmdhw,mkd->bmkhw", x, self._codebook)
        # [b, m, k, h, w], pairwise L2-distance
        distance = x2 + c2 - 2 * inter
        # [b, m, k, h, w], distance as logits to sample
        sample = F.gumbel_softmax(distance, t, hard=True, dim=2)
        # [b, m, d, h, w], use sample to find codewords
        quantized = torch.einsum("bmkhw,mkd->bmdhw", sample, self._codebook)
        # back to [b, n, h, w]
        quantized = quantized.reshape(b, -1, h, w)
        # [b, n, h, w], [b, m, h, w], quantizeds and binaries
        return quantized, sample.argmax(2)
```



# Try Me!
It is easy (with a GPU) to try our model. We would give a quick guide to help you compress an image and restore it.


## Requirements
To run the model, your device needs to meet following requirements.

* Hardware
  * a CUDA-enabled GPU (Driver version `≥ 450.80.02`)
  * `≥ 8GiB RAM`
  * `≥ 8GiB VRAM`
* OS
  * Tested on `Linux`

## Docker (Recommended)
We recommend you to use our pre-built [`docker` images](localhost) to get away from environment issues.

Test with the latest docker image:
```bash
sh -c "$(curl -fsSL https://raw.github.com/xiaosu-zhu/main/docker/demo.sh)"
```
The demo would let you choose an image to compress.

The latest docker image could be accessed by tag: `mcquic/main:latest`.

## Install via PyPI
Another way needs any `conda` environments installed, *e.g.* [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

And instructions are still yet simple.

* Create a virtual env `mcquic` and install all packages
```bash
sh -c "$(curl -fsSL https://raw.github.com/xiaosu-zhu/main/get-mcquic.sh)"
```
* Compress images
```bash
mcquic --help
mcquic -q 1 -i assets/example.png -o assets/compressed.bin
```
* Decompress images
```bash
mcquic -q 1 -i assets/compressed.bin -o assets/example.png
```

## Install Manually (for dev)
This way enables your fully access to codes. Also, `conda` environments are needed, *e.g.* [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

* Clone this repository
```bash
git clone https://github.com/xiaosu-zhu/McQuic.git && cd McQuic
```
* Create a virtual env `mcquic` and install all required packages
```bash
conda env create -f environment.yml
```
* Activate the new environment
```bash
conda activate mcquic
```
* Install this package via `PyPI`
```bash
pip install -e ./
```
* Compress images
```bash
python -m mcquic --help
python -m mcquic -q 1 -i assets/example.png -o assets/compressed.bin
```
* Decompress images
```bash
python -m mcquic -q 1 -i assets/compressed.bin -o assets/example.png
```
And check outputs: [`assets/compressed.bin`](./assets/compressed.bin) and [`assets/restored.png`](./assets/restored.png).

* (***Optional***) Install `NVIDIA/Apex`

[`NVIDIA/Apex`](https://github.com/NVIDIA/apex) is an additional package **required** for training. If you want to [**Develop, contribute, or train a new model**](#develop-contribute-or-train-a-new-model), please ensure you've installed `NVIDIA/Apex`.
```bash
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
More information such as building toolchains, please refer to [their repository](https://github.com/NVIDIA/apex).



# Develop, Contribute, or Train a New Model
It will be very nice 😊 if you want to check your new ideas, add new functions, or train new models, you need to install `mcquic` by [**Docker**](#docker-recommended) or [**manually (with optional step)**](#install-manually-for-dev). To train models, here are minimal and recommended system requirements.

## Requirements
* Minimal
  * `RAM ≥ 16GiB`
  * `VRAM ≥ 12GiB`
* Recommended
  * `VRAM ≥ 24GiB`
  * Better if you have `≥4-way` NVIDIA RTX 3090s or faster GPUs.

## Configs
Files in [configs](configs) give some example configs to train models. Please check specifications in [configs/README.md](configs/README.md).

## Train and Test
Before training models, you need to prepare an image dataset. It is free to pick any images to form dataset, as long as the image-size is `≥512x512`.

* To build a training dataset, please put all images in a folder (allow for sub-folders), then
```bash
# python -m mcquic.dataset [PATH_OF_YOUR_IMAGE_FOLDER] [PATH_OF_OUTPUT_DATASET]
python -m mcquic.dataset train_images mcquic_dataset
```
to build a `lmdb` for `mcquic` to read.

* Then, you could prepare a training config, and don't forget to speify dataset path.
```yaml
# `configs/train.yaml`
...
dataset: mcquic_dataset # path to the training dataset
valDataset: val_images # path to folder of validation images
...
```
where `dataset` and `valDataset` can be any relative or absolute paths.

In this example, the final folder structure is shown below:

```bash
.
... # other files
├─ 📂configs
│   ...
│   └── 📄train.yaml
├── 📂mcquic
├── 📄README.md # this readme
├── 📂train_images # a lot of training images
│   ├── 📂ImageNet
│   |   ├── 📂folder1 # a lot of images
│   |   ├── 🖼️image1.png
│   |   ...
│   ├── 📂COCO
│   |   ├── 🖼️image1.png
│   |   ├── 🖼️image2.png
│   |   ...
|   ...
├── 📂mcquic_dataset # generated training dataset
|   ├── 📀data.mdb
|   ├── 📀lock.mdb
|   └── 📄metadata.json
└── 📂val_images # a lot of validation images
    ├── 🖼️image1.png
    ├── 🖼️image2.png
    ...
```
* To train a new model, run
```bash
python -m mcquic.train --help
# python -O -m mcquic.train -c [PATH_TO_CONFIG]
python -O -m mcquic.train -c configs/train.config
```
Saved model is located in `saved/mcquic_dataset/latest`.
* To resume an interuptted training, run
```bash
python -O -m mcquic.train -r
```
or
```bash
python -O -m mcquic.train -c configs/train.config -r
```
if you want to use a new config (e.g. tuned learning rate, modified hyper-parameters) to resume training.


# To-do List

# Reference and License
