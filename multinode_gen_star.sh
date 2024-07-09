#!/bin/bash

#SBATCH -J mcquic_stage_2
#SBATCH -p ai
#SBATCH -N 5
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=48
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

# ntask should be equal to N

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/GLOBALFS/uestc_jksong_1/McQuic"

module load CUDA/12.2
# source /ssdfs/datahome/tj24011/software/miniconda3/etc/profile.d/conda.sh
# conda activate mcquic

# Graceful restart = 3, for handling data issue
WANDB_MODE=offline NCCL_ALGO=Tree TOKENIZERS_PARALLELISM=false NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 srun torchrun \
--nnodes 5 \
--max_restarts 3 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $HOSTNAME:19936 \
mcquic/train/__main__.py -G configs/imagenet_mcq.yaml
