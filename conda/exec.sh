#!/bin/bash

set -ex
set -o pipefail

conda install pybind11

conda build -c conda-forge -c bioconda -c pytorch -c xiaosu-zhu --output-folder . .

$CONDA/bin/anaconda upload --label main linux-64/*.tar.bz2
