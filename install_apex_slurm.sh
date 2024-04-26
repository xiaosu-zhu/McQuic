#!/bin/bash

#SBATCH -J install_apex
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a800:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Apex is not compatible with cuda 12.1, exiting"
exit

module load cuda/12.1
source /share/home/tj24011/software/miniconda3/etc/profile.d/conda.sh
conda activate mcquic

cd /share/home/tj24011/workspace/apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
