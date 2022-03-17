<p align="center">
  <a href="https://github.com/xiaosu-zhu/McQuic#gh-light-mode-only">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/McQuic-light.svg#gh-light-mode-only" alt="McQuic" title="McQuic" width="45%"/>
  </a>
  <a href="https://github.com/xiaosu-zhu/McQuic#gh-dark-mode-only">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/McQuic-dark.svg#gh-dark-mode-only" alt="McQuic" title="McQuic" width="45%"/>
  </a>
  <br/>
  <span>
    <i>a.k.a.</i> <b><i>M</i></b>ulti-<b><i>c</i></b>odebook <b><i>Qu</i></b>antizers for neural <b><i>i</i></b>mage <b><i>c</i></b>ompression
  </span>
</p>


<p align="center">
  <a href="https://www.python.org/" target="_blank">
    <image src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  </a>
  <a href="https://pytorch.org/" target="_blank">
    <image src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
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


<br/>

<p align="center">
  <b>🥳Our paper will be presented at CVPR 2022!🥳</b>
</p>
<br/>
<p align="center">
  <a href="localhost#gh-light-mode-only">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/papertitle-light.svg#gh-light-mode-only" alt="Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression" title="Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression" width="100%"/>
  </a>
  <a href="localhost#gh-dark-mode-only">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/papertitle-dark.svg#gh-dark-mode-only" alt="Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression" title="Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression" width="100%"/>
  </a>
</p>
  <p align="center"><a href="localhost">CVF Open Access</a> | <a href="localhost">arXiv</a> | <a href="#citation">BibTex</a> | Demo will be online soon</p>



<br/>
<br/>
<br/>

**Mc*****Quic*** is a deep image compressor.

**Features**:
* Solid performance and super-fast coding speed (See [Reference Models](#reference-models)).
* Cross-platform support (Linux-64, Windows-64 and macOS-64, macOS-arm64).

**Techs**:

The **Mc*****Quic*** hold rich multi-codebooks to quantize visual features and restore images by these quantized features. Similar ideas are presented in SHA [[1](#SHA)], VQ-VAE [[2](#VQ-VAE)], VQ-GAN [[3](#VQ-GAN)], *etc*. We summarize these as vectorized priors, and our method extends these ideas to a ***unified multivariate Gaussian mixture***, to perform high-quality, low-latency image compression.

<p align="center">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/priors-light.svg#gh-light-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/priors-dark.svg#gh-dark-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <span><b>Figure 1. Operational diagrams of different methods.</b></span>
</p>


<p align="center">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/kodim24-light.png#gh-light-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/kodim24-dark.png#gh-dark-mode-only" alt="kodim24.png" title="kodim24.png" width="100%">
    <span><b>Figure 2. Comparisons with traditional codecs on an image from Kodak dataset.</b></span>
</p>

<!--ts-->
* [Quick Start](#quick-start)
   * [Requirements](#requirements)
   * [Conda (Recommended)](#conda-recommended)
   * [Docker](#docker)
   * [Install Manually (for dev)](#install-manually-for-dev)
   * [(<em><strong>Optional</strong></em>) Install NVIDIA/Apex](#optional-install-nvidiaapex)
* [Reference Models](#reference-models)
* [Train a New Model](#train-a-new-model)
   * [Requirements](#requirements-1)
   * [Configs](#configs)
   * [Prepare a Dataset](#prepare-a-dataset)
   * [Training](#training)
   * [Test](#test)
* [Implement MCQ by yourself](#implement-mcq-by-yourself)
* [Contribute to this Repository](#contribute-to-this-repository)
* [To-do List](#to-do-list)
* [Detailed framework](#detailed-framework)
* [References and License](#references-and-license)
   * [References](#references)
   * [Citation](#citation)
   * [Copyright](#copyright)

<!-- Added by: runner, at: Thu Mar 17 02:34:20 UTC 2022 -->

<!--te-->



# Quick Start
It is easy (with a GPU, or CPU if you like) to try our model. I would give a quick guide to help you compress an image and restore it.

## Requirements
To run the model, your device needs to meet following requirements.

* Hardware
  * a CUDA-enabled GPU (`≥ 8GiB VRAM`, Driver version `≥ 450.80.02`)
  * If you don't have GPU, running models on CPU may be slower.
  * `≥ 8GiB RAM`
* OS
  * I've tested all features on `Ubuntu`, other platforms should also work. If not, please [file bugs](#contribute-to-this-repository).

## Conda (Recommended)
Install this package is very easy with a `conda` environment installed, *e.g.* [Miniconda](https://docs.conda.io/en/latest/miniconda.html). I recommend you to install it to a new virtual environment directly by:
```bash
# Install a clean pytorch with CUDA support
conda create -n [ENV_NAME] pytorch torchvision cudatoolkit -c pytorch
# Install mcquic and other dependencies
conda install -n [ENV_NAME] mcquic -c xiaosu-zhu -c conda-forge
conda activate [ENV_NAME]
```

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> Above command install packages with `CUDA` support. If you just want to run it on CPU, please use `cpuonly` other than `cudatoolkit` in the first command.

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> Since there is no proper version of torchvision now for Apple M1, you need to change channel from `pytorch` to `conda-forge` in the first command.


* Compress images
```bash
mcquic --help
mcquic -qp 3 path/to/an/image path/to/output.mcq
```
* Decompress images
```bash
# `-qp` is not necessary. Since this arg is written to `output.mcq`.
mcquic path/to/output.mcq path/to/restored.png
```


## Docker
I also build [`docker` images](https://github.com/xiaosu-zhu/McQuic/pkgs/container/mcquic) for you to get away from environment issues.

Try with the latest docker image:
```bash
docker pull ghcr.io/xiaosu-zhu/mcquic:main
```


## Install Manually (for dev)
This way enables your full access to this repo for modifying. Also, if you want to go on, a `conda` environment is needed, *e.g.* [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

* Clone this repository
```bash
git clone https://github.com/xiaosu-zhu/McQuic.git && cd McQuic
```
* Create a virtual env `mcquic` and install all packages by
```powershell
./install.sh  # for POSIX with bash
.\install.ps1 # for Windows with Anaconda PowerShell
```

Now you should in the `mcquic` virtual environment. If not, please activate it by `conda activate mcquic`.

* Compress images
```bash
mcquic --help
mcquic -qp 3 assets/sample.png assets/compressed.mcq
```
* Decompress images
```bash
# `-qp` is not necessary. Since this arg is written to `output.mcq`.
mcquic assets/compressed.mcq assets/restored.png
```
And check outputs: [`assets/compressed.mcq`](https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/compressed.mcq) and [`assets/restored.png`](https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/restored.png).

## (***Optional***) Install `NVIDIA/Apex`

[`NVIDIA/Apex`](https://github.com/NVIDIA/apex) is an additional package **required** for training. If you want to [**develop, contribute**](#contribute-to-this-repository), or [**train a new model**](#train-a-new-model), please ensure you've installed `NVIDIA/Apex` by following snippets.
```bash
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> If you are using [Docker images](#docker), this step is not necessary.

<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> Please make sure you've installed it in the correct virtual environment.


<a href="#">
  <image src="https://img.shields.io/badge/NOTE-yellow?style=for-the-badge" alt="NOTE"/>
</a>

> For more information such as building toolchains, please refer to [their repository](https://github.com/NVIDIA/apex).


# Reference Models
I've released one pretrained model (Sorry, currently I don't have much free GPUs). You could fetch them by specifying `-qp [Model_NO]`. Following is the pretrained model list (Others ***TBA***):


| Model No. 	| Channel 	| M 	|        K        	| Throughput (Encode/Decode) 	| Avg.BPP 	|
|:---------:	|:-------:	|:-:	|:---------------:	|:--------------------------:	|:-------:	|
|         - 	|     -   	| - 	|               - 	|              -             	|    -    	|
|         3 	|   128   	| 2 	| [8192,2048,512] 	|   25.45 Mpps / 22.03 Mpps  	|  0.1277 	|
|         - 	|     -   	| - 	|               - 	|              -             	|    -    	|

The coding throughput is tested on a NVIDIA RTX 3090. Image file I/O, loading and other operations are not included in the test.
- **`Mpps = Mega-pixels per second`**
- **`BPP = Bits per pixel`**

# Train a New Model
Please ensure you've installed [`NVIDIA/Apex`](#optional-install-nvidiaapex). To train models, here are minimal and recommended system requirements.

## Requirements
* Minimal
  * `RAM ≥ 16GiB`
  * `VRAM ≥ 12GiB`
* Recommended
  * `VRAM ≥ 24GiB`
  * Better if you have `≥4-way` NVIDIA RTX 3090s or faster GPUs.

## Configs
The folder [configs](configs) provides example config `example.yaml` to train models. Please check specifications in [configs/README.md](configs/README.md).

## Prepare a Dataset
Before training models, you need to prepare an image dataset. It is free to pick any images to form dataset, as long as the image-size is `≥512x512`.

* To build a training dataset, please put all images in a folder (allow for sub-folders), then run
```bash
mcquic dataset --help
# mcquic dataset [PATH_OF_YOUR_IMAGE_FOLDER] [PATH_OF_OUTPUT_DATASET]
mcquic dataset train_images mcquic_dataset
```
to build a `lmdb` dataset for `mcquic` to read.

* Then, you could prepare a training config *e.g.* `configs/train.yaml`, and don't forget to speify dataset path.
```yaml
# `configs/train.yaml`
...
trainSet: mcquic_dataset # path to the training dataset.
valSet: val_images # path to a folder of validation images.
savePath: saved # path to a folder to save checkpoints.
...
```
where `trainSet` and `valSet` can be any relative or absolute paths, and `savePath` is a folder for saving checkpoints and logs.

In this example, the final folder structure is shown below:

```yaml
. # A nice folder
├─ 📂configs
│   ...
│   └── 📄train.yaml
├── 📄README.md # this readme
├── 📂saved # saved models apprear here
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

## Training
* To train a new model, run
```bash
mcquic train --help
# mcquic train [PATH_TO_CONFIG]
mcquic train configs/train.yaml
```
and saved model is located in `saved/mcquic_dataset/latest`.
* To resume an interuptted training, run
```bash
mcquic train -r
```
, or
```bash
mcquic train -r configs/train.yaml
```
if you want to use an updated config (e.g. tuned learning rate, modified hyper-parameters) to resume training.


## Test
You could use any save checkpoints (usually located in above `savePath`) to validate the performance. For example
```bash
mcquic validate --help
mcquic validate path/to/a/checkpoint path/to/images/folder path/to/final/model
```

And the output "final model" is compatible with the main program `mcquic`, you could directly use this local model to perform compression. Try:
```bash
mcquic --local assets/sample.png assets/compressed.mcq
# `--local` is not necessary. Since this arg is written to `output.mcq`.
mcquic assets/compressed.mcq assets/restored.png
```
If you think your model is awesome, please don't hasitate to [Contribute to this Repository](#contribute-to-this-repository)!



# Implement MCQ by yourself
A minimal implementation of the multi-codebook quantizer comes up with:

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


# Contribute to this Repository
It will be very nice if you want to check your new ideas or add new functions 😊. You will need to install `mcquic` by [**Docker**](#docker-recommended) or [**manually (with optional step)**](#install-manually-for-dev). Just like other git repos, before raising issues or pull requests, please take a thorough look at [issue templates](https://github.com/xiaosu-zhu/McQuic/issues/new/choose).


# To-do List
* Docker file and bash script
* execute pattern: `mcquic service`
* reference model: 1,2,4~4+
* change compressor args

# Detailed framework
Thanks for your attention!❤️ Here are details in the paper.

Following previous works, we build the compression model as an AutoEncoder. Bottleneck of encoder (analysis transform) outputs a small feature map and is quantized by *multi-codebook vector-quantization* other than scalar-quantization. Quantizers are cascaded to effectively estimate latent distribution.

<p align="center">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/framework-light.svg#gh-light-mode-only" alt="Framework" title="Framework" width="100%">
    <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/paper/framework-dark.svg#gh-dark-mode-only" alt="Framework" title="Framework" width="100%">
    <span><b>Figure 3. Left: Overall framework. Right: Structure of a quantizer.</b></span>
</p>

Right part of above figure shows detailed structure of our proposed quantizer.

# References and License
## References
[<a id="SHA">1</a>] Agustsson, Eirikur, et al. "Soft-to-hard vector quantization for end-to-end learning compressible representations." NeurIPS 2017.

[<a id="VQ-VAE">2</a>] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." NeurIPS 2017.

[<a id="VQ-GAN">3</a>] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." CVPR 2021.

## Citation
To cite our paper, please use following BibTex:
```plain
@inproceedings{McQuic,
  author    = {Xiaosu Zhu and
               Jingkuan Song and
               Lianli Gao and
               Feng Zheng and
               Heng Tao Shen},
  title     = {Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression},
  booktitle = {CVPR},
  % pages     = {????--????}
  year      = {2022}
}
```

## Copyright

**Fonts**:
* [**Source Sans Pro**](https://fonts.adobe.com/fonts/source-sans). © 2010, 2012 Adobe Systems Incorporated, SIL Open Font License.
* [**Flash Rogers 3D**](https://www.iconian.com/index.html). © 2007 Iconian Fonts, donationware.
* [**Cambria Math**](https://docs.microsoft.com/en-us/typography/font-list/cambria-math). © 2017 Microsoft Corporation. All rights reserved.
* [**Times New Roman**](https://docs.microsoft.com/en-us/typography/font-list/times-new-roman). © 2017 The Monotype Corporation. All Rights Reserved.
* [**Caramel and Vanilla**](http://www.foundmyfont.com/). © 2017 FOUND MY FONT LTD. All Rights Reserved.

**Pictures**:
* [**kodim24.png**](http://r0k.us/graphics/kodak/kodim24.html) by Alfons Rudolph, Kodak Image Dataset.
* [**assets/sample.png**](https://unsplash.com/photos/hLxqYJspAkE) by Ales Krivec, CLIC Professional valid set.


**Third-party repos**:

| Repos                                                                          | License |
|-------------------------------------------------------------------------------:|---------|
| [PyTorch](https://pytorch.org/)                                                | [BSD-style](https://github.com/pytorch/pytorch/blob/master/LICENSE) |
| [Torchvision](https://pytorch.org/vision/stable/index.html)                    | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [Apex](https://nvidia.github.io/apex/)                                         | [BSD-3-Clause](https://github.com/NVIDIA/apex/blob/master/LICENSE) |
| [tqdm](https://tqdm.github.io/)                                                | [MPLv2.0, MIT](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| [Tensorboard](https://www.tensorflow.org/tensorboard)                          | [Apache-2.0](https://github.com/tensorflow/tensorboard/blob/master/LICENSE) |
| [rich](https://rich.readthedocs.io/en/latest/)                                 | [MIT](https://github.com/Textualize/rich/blob/master/LICENSE) |
| [python-lmdb](https://lmdb.readthedocs.io/en/release/)                         | [OpenLDAP Version 2.8](https://github.com/jnwatson/py-lmdb/blob/master/LICENSE) |
| [PyYAML](https://pyyaml.org/)                                                  | [MIT](https://github.com/yaml/pyyaml/blob/master/LICENSE) |
| [marshmallow](https://marshmallow.readthedocs.io/en/stable/)                   | [MIT](https://github.com/marshmallow-code/marshmallow/blob/dev/LICENSE) |
| [click](https://click.palletsprojects.com/)                                    | [BSD-3-Clause](https://github.com/pallets/click/blob/main/LICENSE.rst) |
| [vlutils](https://github.com/VL-Group/vlutils)                                 | [Apache-2.0](https://github.com/VL-Group/vlutils/blob/main/LICENSE) |
| [MessagePack](https://msgpack.org/)                                            | [Apache-2.0](https://github.com/msgpack/msgpack-python/blob/main/COPYING) |
| [pybind11](https://pybind11.readthedocs.io/en/stable/)                         | [BSD-style](https://github.com/pybind/pybind11/blob/master/LICENSE) |
| [CompressAI](https://interdigitalinc.github.io/CompressAI/)                    | [BSD 3-Clause Clear](https://github.com/InterDigitalInc/CompressAI/blob/master/LICENSE) |
| [Taming-transformer](https://compvis.github.io/taming-transformers/)           | [MIT](https://github.com/CompVis/taming-transformers/blob/master/License.txt) |
| [marshmallow-jsonschema](https://github.com/fuhrysteve/marshmallow-jsonschema) | [MIT](https://github.com/fuhrysteve/marshmallow-jsonschema/blob/master/LICENSE) |
| [json-schema-for-humans](https://coveooss.github.io/json-schema-for-humans/#/) | [Apache-2.0](https://github.com/coveooss/json-schema-for-humans/blob/main/LICENSE.md) |
| [CyclicLR](https://github.com/bckenstler/CLR)                                  | [MIT](https://github.com/bckenstler/CLR/blob/master/LICENSE) |
| [batch-transforms](https://github.com/pratogab/batch-transforms)               | [MIT](https://github.com/pratogab/batch-transforms/blob/master/LICENSE) |
| [pytorch-msssim](https://github.com/VainF/pytorch-msssim) | [MIT](https://github.com/VainF/pytorch-msssim/blob/master/LICENSE) |
| [Streamlit](https://streamlit.io/) | [Apache-2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE) |


<br/>
<br/>
<p align="center">
<b>
This repo is licensed under
</b>
</p>
<p align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0#gh-light-mode-only" target="_blank">
  <img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/ASF_Logo-light.svg#gh-light-mode-only" alt="The Apache Software Foundation" title="The Apache Software Foundation" width="200px"/>
</a>
<a href="https://www.apache.org/licenses/LICENSE-2.0#gh-dark-mode-only" target="_blank">
<img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/ASF_Logo-light.svg#gh-dark-mode-only" alt="The Apache Software Foundation" title="The Apache Software Foundation" width="200px"/>
</a>
</p>
<p align="center">
<a href="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/LICENSE">
  <b>Apache License<br/>Version 2.0</b>
</a>
</p>

<br/>
<br/>
<br/>

<p align="center">
<a href="https://github.com/yaya-cheng#gh-dark-mode-only">
<img src="https://raw.githubusercontent.com/xiaosu-zhu/McQuic/main/assets/thanks.svg#gh-dark-mode-only" width="250px"/>
</a>
</p>
