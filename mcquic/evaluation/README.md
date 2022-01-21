```console
CUDA_VISIBLE_DEVICES=4 python -O src/mcquic/evaluation/scripting.py --cfg configs/ssim/v100/2-13+11+9.yaml --path saved/fullDataset/latest/best.ckpt --target ckpt/

CUDA_VISIBLE_DEVICES=4 python -O src/mcquic/evaluation/eval.py --cfg configs/ssim/v100/2-13+11+9.yaml --encoder ckpt/encoder.pt --decoder ckpt/decoder.pt
```
