#!/bin/bash

#SBATCH -J mcquic_pretraining
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

# ntask should be equal to N

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="/ssdfs/datahome/tj24011/workspace/McQuic"

module load cuda/12.1
source /ssdfs/datahome/tj24011/software/miniconda3/etc/profile.d/conda.sh
conda activate mcquic

# Graceful restart = 3, for handling data issue
TOKENIZERS_PARALLELISM=false srun python /ssdfs/datahome/tj24011/workspace/McQuic/mcquic/train/__main__.py /ssdfs/datahome/tj24011/workspace/McQuic/configs/a800_16.yaml
