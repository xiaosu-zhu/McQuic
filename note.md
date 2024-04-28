添加 `NCCL_P2P_LEVEL=NVL` 环境变量提升显卡通信效率

quantizer 部分去除了 einsum，einsum 会导致 grad stride 与 bucket view 不一致

由于level太多，rans coder不能用


编译 apex：

1. gcc 9.4, gxx 9.4 (conda)
2. srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua01 --pty /bin/bash
3. module load cuda/12.1
4. MAX_JOBS=32 pip install ...


新运行命令：
```
*********     不要使用 python 3.12     ************
********* 这会导致 torchrun seg fault  ************
https://github.com/pytorch/pytorch/issues/116423
```
```bash
# IF You don't want to resume training!
rm -rf saved/latest
TOKENIZERS_PARALLELISM=false NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 mcquic/train/__main__.py configs/neon.yaml
```

```bash
# process dataset
srun -N 1 -p amd -n 1 --cpus-per-task=32 --nodelist=cpua05 -J openimage_clean_create /ssdfs/datahome/tj24011/software/miniconda3/envs/mcquic/bin/mcquic dataset /ssdfs/datahome/tj24011/datasets/raw/openimages/ /ssdfs/datahome/tj24011/datasets/webdataset/openimages_HQ/
```

使用 slurm 时：
```bash
sbatch singlenode.sh
# 输出在 slurm-jobid.out, slurm-jobid.err
# job 状态
squeue --job JOB_ID
# 取消
scancel JOB_ID
```


如果多节点出现： The server socket has failed to listen on any local network address

The server socket has failed to bind to [::]:19936

是因为 ntask 比 N 多，要设置 ntask == N，这样一个 node 刚好启动一个 torchrun，不然后面每个 node 有多个 torchrun 冲突了
