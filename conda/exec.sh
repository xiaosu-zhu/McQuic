#!/bin/bash

set -ex
set -o pipefail

conda config --set anaconda_upload yes

conda build -c conda-forge -c bioconda -c pytorch -c xiaosu-zhu --output-folder build conda/
