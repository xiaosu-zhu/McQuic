# Tested under this image, try newer images on your own risk
FROM nvcr.io/nvidia/pytorch:21.07-py3

COPY environment.yaml ./

RUN conda env update --name base --file environment.yaml --prune

COPY ../../mcqc ./

RUN cd mcqc/src && pip install -e .
