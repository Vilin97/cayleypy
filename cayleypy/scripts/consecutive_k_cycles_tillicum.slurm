#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

GEN_FAMILIES=(consecutive wrapped)

k=4
n=36
tid=$SLURM_ARRAY_TASK_ID

generator_family=${GEN_FAMILIES[$tid]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --generator_family "${generator_family}" \
  --central_mode alternating \
  --device cuda