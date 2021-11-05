# Tested under this image, try newer images on your own risk
FROM nvcr.io/nvidia/pytorch:21.09-py3

COPY environment.yaml ./

RUN conda env update --name base --file environment.yml --prune

RUN rm environment.yaml

COPY ../../mcqc ./

RUN cd mcqc/src && pip install -e .

WORKDIR /workspace/mcqc
