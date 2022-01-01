# Tested under this image, try newer images at your own risk
FROM nvcr.io/nvidia/pytorch:21.12-py3

COPY environment.yml ./

RUN conda env update --name base --file environment.yml --prune

RUN rm environment.yml

RUN git clone https://github.com/xiaosu-zhu/mcqc.git && cd mcqc/src && pip install -e .

RUN cd mcqc && ln -s /workspace/datasets data

# COPY ../../mcqc ./

# RUN cd mcqc/src && pip install -e .

WORKDIR /workspace/mcqc
