# mcqc
docker run -u $(id -u):$(id -g) --gpus all --rm -it --ipc=host -v /raid/zhuxiaosu/codes/mcqc:/workspace/mcqc -v /raid/zhuxiaosu/datasets:/workspace/mcqc/data zhongbazhu/mcqc:base /bin/bash

# tensorflow-gpu
docker run -u $(id -u):$(id -g) --gpus all --rm -it --ipc=host -v /raid/zhuxiaosu/codes/mcqc:/workspace/mcqc -v /raid/zhuxiaosu/datasets:/workspace/mcqc/data tensorflow/tensorflow:2.5.1-gpu /bin/bash
