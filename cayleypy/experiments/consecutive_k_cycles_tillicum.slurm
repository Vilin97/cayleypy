#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(7)

NS=(34 35 36 37 38)

tid=$SLURM_ARRAY_TASK_ID

idx_k=$(( tid % 1 ))
idx_n=$(( tid / 1 ))

k=${KS[$idx_k]}
n=${NS[$idx_n]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}"
