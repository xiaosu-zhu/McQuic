# Tested under this image, try newer images on your own risk
FROM nvcr.io/nvidia/pytorch:21.12-py3

COPY environment.yml ./

RUN conda env update --name base --file environment.yml --prune

RUN rm environment.yml

COPY ../../mcqc ./

RUN cd mcqc/src && pip install -e .

WORKDIR /workspace/mcqc
