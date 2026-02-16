#!/bin/bash
#SBATCH --job-name=k10_n38_4gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%N_%A_%a.out

# k=10, n=38, consecutive, not inverse closed, 4 GPUs on tillicum.

NUM_GPUS=4
K=10
N=38

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${K}" \
  --n "${N}" \
  --device cuda \
  --generator_family consecutive \
  --coset True \
  --central_mode block \
  --inverse_closed False \
  --num_gpus "${NUM_GPUS}"
