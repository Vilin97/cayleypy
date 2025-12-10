#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(2 3 4)
NS=(30 31 32 33 34 35)

GENERATOR_FAMILY=consecutive
CENTRAL_MODE=alternating
DEVICE=cuda

tid=$SLURM_ARRAY_TASK_ID

num_n=${#NS[@]}          # 6
idx_k=$(( tid / num_n )) # 0..2
idx_n=$(( tid % num_n )) # 0..5

k=${KS[$idx_k]}
n=${NS[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --central_mode "${CENTRAL_MODE}"
