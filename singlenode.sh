#!/bin/bash

#SBATCH -J mcquic_pretraining
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a800:8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load cuda/12.1
source /share/home/tj24011/software/miniconda3/etc/profile.d/conda.sh
conda activate mcquic

NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=8 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=8 /share/home/tj24011/workspace/McQuic/mcquic/train/__main__.py /share/home/tj24011/workspace/McQuic/configs/a800_8.yaml
