#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(2 3 4 5 8 9 10 11 14 15)
NS=(37)
GENERATOR_FAMILY=consecutive
CENTRAL_MODE=alternating
DEVICE=cpu

tid=$SLURM_ARRAY_TASK_ID

idx_k=$tid
idx_n=0

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
