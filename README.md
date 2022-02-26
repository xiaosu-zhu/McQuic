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
    <p align="center"><b>Figure 1. Operational diagrams of different methods.</b></figcaption>
</p>


<p align="center">
    <img src="./assets/paper/kodim24-light.png#gh-light-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <img src="./assets/paper/kodim24-dark.png#gh-dark-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <p align="center"><b>Figure 2. Comparisons with traditional codecs on an image from Kodak dataset.</b></figcaption>
</p>

<!--ts-->
* [Introduction](#introduction)
* [Try Me!](#try-me)
   * [Requirements](#requirements)
      * [Docker (Recommended)](#docker-recommended)
      * [Install Manually](#install-manually)
   * [Usage](#usage)
      * [Compress Images](#compress-images)
      * [Decompress Images](#decompress-images)
* [Train a New Model](#train-a-new-model)
   * [Requirements](#requirements-1)
   * [Configs](#configs)
   * [Train and Test](#train-and-test)
* [To-do List](#to-do-list)
* [Reference and License](#reference-and-license)

<!-- Added by: runner, at: Fri Feb 25 12:51:08 UTC 2022 -->

<!--te-->

# Introduction

Following previous works, we build the compression model as an AutoEncoder. Bottleneck of encoder (analysis transform) outputs a small feature map and is quantized by *multi-codebook vector-quantization* other than scalar-quantization. Quantizers are cascaded to effectively estimate latent distribution.

<p align="center">
    <img src="./assets/paper/framework-light.svg#gh-light-mode-only" alt="Framework" title="Framework" width="100%">
    <img src="./assets/paper/framework-dark.svg#gh-dark-mode-only" alt="Framework" title="Framework" width="100%">
    <p align="center"><b>Figure 3. Left: Overall framework. Right: Structure of a quantizer.</b></figcaption>
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

It is easy (with a GPU) to try our model. We will give a quick guide to compress an image and restore it.

## Requirements

To run the model, your device should meet following requirements.

* Hardware
  * `a CUDA-enabled GPU (Driver version ≥ 450.80.02)`
  * `≥ 8GiB RAM`
  * `≥ 8GiB VRAM`
* OS
  * Tested on `Linux`

## Docker (Recommended)

We recommend you to use our pre-built [`docker` images](localhost) to get away from environment issues.

Test with the latest docker image:
```console
./docker/demo
```
And check outputs: [`assets/compressed.bin`](./assets/compressed.bin) and [`assets/restored.png`](./assets/restored.png).

## Install Manually
Please ensure you've installed any `conda` environments, *e.g.* [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

* Clone this repository
```console
git clone https://github.com/xiaosu-zhu/McQuic.git && cd McQuic
```
* Install all required packages with `conda`
```console
conda env create -f environment.yml
```
* Activate the new environment
```console
conda activate mcquic
```
* Install this package via `pip`
```console
pip install -e .
```
* Compress images
```console
python -m mcquic --help
python -m mcquic -q 1 -i assets/example.png -o assets/compressed.bin
```
* Decompress images
```console
python -m mcquic -q 1 -i assets/compressed.bin -o assets/example.png
```
And check outputs: [`assets/compressed.bin`](./assets/compressed.bin) and [`assets/restored.png`](./assets/restored.png).
# Train a New Model

## Requirements

## Configs

## Train and Test

# To-do List

# Reference and License
