<p align="center">
    <img src="./assets/McQuic-light.svg#gh-light-mode-only" alt="McQuic" title="McQuic" width="33%">
    <img src="./assets/McQuic-dark.svg#gh-dark-mode-only" alt="McQuic" title="McQuic" width="33%">
    <br/>
    <p align="center"><i>a.k.a.</i> <b><i>M</i></b>ulti-<b><i>c</i></b>odebook <b><i>Qu</i></b>antizers for neural <b><i>i</i></b>mage <b><i>c</i></b>ompression</p>
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


Multi-codebook quantizers hold rich codebooks to quantize visual features and restore images by these quantized features. Similar ideas represent in SHA [[1](#reference-and-license)], VQ-VAE [[2](#reference-and-license)], VQ-GAN [[3](#reference-and-license)], *etc*. We summarize these as vectorized priors, and our method extends these ideas to a unified multivariate Gaussian mixture, to perform high-quality, low-latency image compression. Take a more look at [**our paper**](localhost).

<p align="center">
    <img src="./assets/paper/priors-light.svg#gh-light-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <img src="./assets/paper/priors-dark.svg#gh-dark-mode-only" alt="Vectorized prior" title="Vectorized prior" width="100%">
    <p align="center"><b>Figure 1. Operational diagrams of different methods.</b></figcaption>
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

<!-- Added by: runner, at: Fri Feb 25 12:22:33 UTC 2022 -->

<!--te-->

# Introduction

# Try Me!

## Requirements

### Docker (Recommended)

### Install Manually

## Usage
### Compress Images

### Decompress Images

# Train a New Model

## Requirements

## Configs

## Train and Test

# To-do List

# Reference and License
