#!/bin/bash

set -ex
set -o pipefail


conda build -c conda-forge -c bioconda -c pytorch -c xiaosu-zhu --output-folder . .

$CONDA/bin/anaconda upload --label main linux-64/*.tar.bz2
