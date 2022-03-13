# Tested under this image, try newer images at your own risk
FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN conda install --only-deps -c xiaosu-zhu -c conda-forge -c pytorch -c anaconda mcquic

WORKDIR /workspace

RUN git clone https://github.com/xiaosu-zhu/mcquic.git && cd mcquic && python setup.py install

WORKDIR /workspace/mcquic
