#!/bin/bash
set -ex


echo "Start installation"

if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please ensure you've installed conda and place it in PATH."
    exit
fi


conda create -y -n mcquic python=3.9 cudatoolkit torchvision pytorch -c pytorch

eval "$(conda shell.$(ps -p $$ -ocomm=) hook)"

conda activate mcquic

conda install -y -n mcquic tqdm pybind11 pip "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "vlutils" "msgpack-python<2" -c xiaosu-zhu -c conda-forge


if [ "$CONDA_DEFAULT_ENV" != "mcquic"]
then
    echo "Can't activate conda env mcquic, exit."
    exit
fi


PYPI_BUILDING=SET pip install -e .


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i "1 s|$| -O|" "$(which mcquic)"*
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i "" "1 s|$| -O|" "$(which mcquic)"*
else
    sed -i "1 s|$| -O|" "$(which mcquic)"*
fi

echo "Installation done!"

echo "If you want to train models, please install NVIDIA/Apex manually."
