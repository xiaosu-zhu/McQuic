# Tested under this image, try newer images at your own risk
FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN conda install "tqdm" "tensorboard<3" "rich<11" "python-lmdb<2" "pyyaml<7" "marshmallow<4" "click<9" "vlutils" "msgpack-python<2" -c xiaosu-zhu -c conda-forge -c pytorch

WORKDIR /workspace

RUN git clone https://github.com/xiaosu-zhu/mcquic.git && cd mcquic && pip install -e .

RUN sed -i "1 s|$| -O|" /opt/conda/bin/mcquic*

WORKDIR /workspace/mcquic
