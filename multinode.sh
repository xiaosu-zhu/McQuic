#!/bin/bash

#SBATCH -J mcquic_pretraining
#SBATCH -p A800
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:a800:8
#SBATCH --cpus-per-task=48
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ntask should be equal to N

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/share/home/tj24011/workspace/McQuic"

module load cuda/12.1
source /share/home/tj24011/software/miniconda3/etc/profile.d/conda.sh
conda activate mcquic


NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=8 srun /share/home/tj24011/software/miniconda3/envs/mcquic/bin/torchrun \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $HOSTNAME:19936 \
/share/home/tj24011/workspace/McQuic/mcquic/train/__main__.py /share/home/tj24011/workspace/McQuic/configs/a800_8.yaml
