#!/bin/bash
set -e
set -o pipefail


echo "Start installation"

if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please ensure you've installed conda and place it in PATH."
    exit
fi


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    conda create -y -n mcquic python=3.9 cudatoolkit "torchvision>=0.12,<1" "pytorch>=1.11,<2" -c pytorch
elif [[ "$OSTYPE" == "darwin"* ]]; then
    conda create -y -n mcquic python=3.9 "torchvision>=0.12,<1" "pytorch>=1.11,<2" -c pytorch
else
    conda create -y -n mcquic python=3.9 cudatoolkit "torchvision>=0.12,<1" "pytorch>=1.11,<2" -c pytorch
fi

eval "$(conda shell.bash hook)"

conda activate mcquic

conda install -y -n mcquic "pybind11>=2.6,<3" "pip>=22" "tensorboard>=2.3,<3" "rich>=10,<11" "python-lmdb>=1.2,<2" "pyyaml>=5.4,<7" "marshmallow>=3.14,<4" "click>=8,<9" "vlutils" "msgpack-python>=1,<2" packaging -c xiaosu-zhu -c conda-forge


if [ "$CONDA_DEFAULT_ENV" != "mcquic"]
then
    echo "Can't activate conda env mcquic, exit."
    exit
fi


ADD_ENTRY=SET pip install -e .


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "1 s|$| -O|" "$(which mcquic)"*
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i "" "1 s|$| -O|" "$(which mcquic)"*
else
    sed -i "1 s|$| -O|" "$(which mcquic)"*
fi

echo "Installation done!"

echo "If you want to train models, please install NVIDIA/Apex manually."
echo "If you want to use streamlit service, please install streamlit via pip."
