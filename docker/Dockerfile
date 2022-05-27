# Tested under this image, try newer images at your own risk
FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN conda install -y "pybind11>=2.6,<3" "pip>=22" "tensorboard>=2.3,<3" "rich>=10,<11" "python-lmdb>=1.2,<2" "pyyaml>=5.4,<7" "marshmallow>=3.14,<4" "click>=8,<9" "msgpack-python>=1,<2" packaging -c conda-forge

WORKDIR /workspace

RUN git clone https://github.com/xiaosu-zhu/mcquic.git && cd mcquic && PYPI_BUILDING="SET" ADD_ENTRY="SET" pip install -e .

RUN sed -i "1 s|$| -O|" "$(which mcquic)"*

WORKDIR /workspace/mcquic

ENTRYPOINT ["mcquic"]
