# Tested under this image, try newer images at your own risk
FROM nvcr.io/nvidia/pytorch:21.12-py3

COPY environment.yml ./

RUN conda env update --name base --file environment.yml --prune

RUN rm environment.yml

RUN git clone https://github.com/xiaosu-zhu/mcquic.git && cd mcquic/src && pip install -e .

RUN cd mcquic && ln -s /workspace/datasets data

# COPY ../../mcquic ./

# RUN cd mcquic/src && pip install -e .

WORKDIR /workspace/mcquic
