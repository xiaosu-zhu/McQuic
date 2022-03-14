#!/bin/bash
set -ex
set -o pipefail


echo "Start installation"

if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please ensure you've installed conda and place it in PATH."
    exit
fi


conda create -y -n mcquic cudatoolkit tqdm pybind11 pip "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "vlutils" "msgpack-python<2" -c xiaosu-zhu -c conda-forge -c pytorch

eval "$(conda shell.bash hook)"

conda activate mcquic

if [ "$CONDA_DEFAULT_ENV" != "mcquic"]
then
    echo "Can't activate conda env mcquic, exit."
    exit
fi

cp setup.cfg setup.cfg.bak

python ci/pre_build/cfg_entry_points.py setup.cfg

pip install -e .

sed -i "1 s|$| -O|" "$(which mcquic)"*

rm -f setup.cfg

mv setup.cfg.bak setup.cfg

echo "Installation done!"
