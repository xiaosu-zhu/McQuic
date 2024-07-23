#!/bin/bash

#SBATCH -J install_apex
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

# echo "Apex is not compatible with cuda 12.1, exiting"
# exit

module load CUDA/12.2
# source /ssdfs/datahome/tj24011/software/miniconda3/etc/profile.d/conda.sh
# conda activate mcquic

cd /GLOBALFS/uestc_jksong_1/software/apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
