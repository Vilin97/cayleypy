#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/%x_%N_%A_%a.out

# k = 5..10 (6 values)
KS=(5 6 7 8 9 10)

# n = 32..36 (5 values)
NS=(32 33 34 35 36)

tid=$SLURM_ARRAY_TASK_ID

# decode 2-D grid index: 6 * 5 = 30 jobs → array 0–29
idx_k=$(( tid % 6 ))
idx_n=$(( tid / 6 ))

k=${KS[$idx_k]}
n=${NS[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}"
