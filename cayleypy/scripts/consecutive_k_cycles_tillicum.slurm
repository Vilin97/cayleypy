#!/bin/bash
#SBATCH --job-name=all_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --array=0-83
#SBATCH --output=logs/%x_%N_%A_%a.out

# Combined job for all configurations:
# 1. consecutive + inverse_closed: k=2-15, n=36,37 (28 jobs: indices 0-27)
# 2. wrapped + inverse_closed: k=2-15, n=36,37 (28 jobs: indices 28-55)
# 3. wrapped + no inverse: k=2-15, n=36,37 (28 jobs: indices 56-83)
# Total: 84 jobs

KS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)
NS=(36 37)

DEVICE=cuda

tid=$SLURM_ARRAY_TASK_ID

# Determine configuration based on task ID
if [ $tid -lt 28 ]; then
    # consecutive + inverse_closed (indices 0-27)
    GENERATOR_FAMILY=consecutive
    INVERSE_CLOSED="--inverse_closed"
    config_tid=$tid
elif [ $tid -lt 56 ]; then
    # wrapped + inverse_closed (indices 28-55)
    GENERATOR_FAMILY=wrapped
    INVERSE_CLOSED="--inverse_closed"
    config_tid=$((tid - 28))
else
    # wrapped + no inverse (indices 56-83)
    GENERATOR_FAMILY=wrapped
    INVERSE_CLOSED=""
    config_tid=$((tid - 56))
fi

# Calculate k and n indices within each configuration
k_idx=$((config_tid / 2))
n_idx=$((config_tid % 2))

k=${KS[$k_idx]}
n=${NS[$n_idx]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --coset \
  --central_mode block \
  ${INVERSE_CLOSED}