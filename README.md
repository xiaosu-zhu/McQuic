`python -O -m mcqc -c CONFIG_PATH`

# McQuic

Here is ***McQuic***, *aka* ***M***ulti-***c***odebook ***Qu***antizers for neural ***i***mage ***c***ompression.

```bash
docker run -u $(id -u):$(id -g) --gpus all --rm -it --ipc=host -v /raid/zhuxiaosu/codes/mcqc:/workspace/mcqc -v /raid/zhuxiaosu/datasets:/workspace/mcqc/data zhongbazhu/mcqc:base /bin/bash
```

# tensorflow-gpu
```bash
docker run -u $(id -u):$(id -g) --gpus all --rm -it --ipc=host -v /raid/zhuxiaosu/codes/mcqc:/workspace/mcqc -v /raid/zhuxiaosu/datasets:/workspace/mcqc/data tensorflow/tensorflow:2.5.1-gpu /bin/bash
```


# bpg encode
change the -m -q option

```bash
bpgenc -m 4 -q 47 data/datasets/kodak/kodim01.png -o kodim01.bpg
```

4+47 is around 0.1270


# bpg decode
```bash
bpgdec kodim01.bpg -o bpg_kodim01.png
```

# vvc
convert to yuv
```bash
ffmpeg -i data/datasets/kodak/kodim01.png -pix_fmt yuv444p kodim01.yuv
```
change the -q option
```bash
VVC/EncoderApp -c VVC/cfg/encoder_intra_vtm.cfg -i kodim01.yuv -b kodim01.bin -o out.yuv -f 1 -fr 2 -wdt 768 -hgt 512 -q 32 --OutputBitDepth=8 --OutputBitDepthC=8 --InputChromaFormat=444
```

convert back to png
```bash
ffmpeg -i out.yuv -pix_fmt yuv444p vvc_kodim01.png
```
