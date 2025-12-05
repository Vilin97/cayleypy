#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(6 8 9 10 11 12 13 14 15)
MODES=(consecutive wrapped)

n=36
tid=$SLURM_ARRAY_TASK_ID

num_modes=${#MODES[@]}  # 2
idx_k=$(( tid / num_modes ))
idx_m=$(( tid % num_modes ))

k=${KS[$idx_k]}
mode=${MODES[$idx_m]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --mode "${mode}"
