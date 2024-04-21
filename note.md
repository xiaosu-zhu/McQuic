添加 `NCCL_P2P_LEVEL=NVL` 环境变量提升显卡通信效率

quantizer 部分去除了 einsum，einsum 会导致 grad stride 与 bucket view 不一致

由于level太多，rans coder不能用


新运行命令：
```
*********     不要使用 python 3.12     ************
********* 这会导致 torchrun seg fault  ************
https://github.com/pytorch/pytorch/issues/116423
```
```bash
NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 mcquic/train/__main__.py configs/neon.yaml
```

使用 slurm 时：
```bash
sbatch sbatch_run.sh
```
