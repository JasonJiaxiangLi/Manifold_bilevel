#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=mani_bilevel
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=mhong

# Benchmark info
echo "TIMING - Starting running at: $(date)"

export CONDA_ENVS_PATH="/home/mhong/li003755/.conda/envs"
source activate spin
nvidia-smi
which python3
echo "Job is starting on $(hostname)"

cd /home/mhong/li003755/Manifold_bilevel || exit

python meta_learning_tests/test_meta_learning.py

exit