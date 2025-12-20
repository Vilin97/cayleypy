#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

K=6
NS=(12 13 14)

GENERATOR_FAMILY=consecutive
DEVICE=cuda

tid=$SLURM_ARRAY_TASK_ID
n=${NS[$tid]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${K}" \
  --n "${n}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}"
