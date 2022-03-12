#!/bin/bash

set -ex
set -o pipefail

conda install pybind11

conda config --set anaconda_upload yes

conda build -c conda-forge -c bioconda -c pytorch -c xiaosu-zhu --output-folder . .
