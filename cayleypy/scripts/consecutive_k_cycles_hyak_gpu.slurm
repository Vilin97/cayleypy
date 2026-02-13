#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-111
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)
NS=(28 29 30 31 32 33 34 35)

GENERATOR_FAMILY=consecutive
DEVICE=cuda

tid=$SLURM_ARRAY_TASK_ID
num_n=${#NS[@]}
k_idx=$((tid / num_n))
n_idx=$((tid % num_n))
k=${KS[$k_idx]}
n=${NS[$n_idx]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --coset True \
  --central_mode block \
  --inverse_closed False
