#!/bin/bash

#SBATCH -J mcquic_stage_2
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=48
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

# ntask should be equal to N

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/ssdfs/datahome/tj24011/workspace/McQuic"

module load cuda/12.1
source /ssdfs/datahome/tj24011/software/miniconda3/etc/profile.d/conda.sh
conda activate mcquic

# Graceful restart = 3, for handling data issue
TOKENIZERS_PARALLELISM=false NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 srun torchrun \
--nnodes 1 \
--max_restarts 3 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint localhost:0 \
mcquic/train/__main__.py -G configs/journey_mcq.yaml