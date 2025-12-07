#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=15
#SBATCH --mem=170G
#SBATCH --time=41:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(4 5 6)
NS=(37)
MODES=(consecutive)

tid=$SLURM_ARRAY_TASK_ID

idx_k=$tid
idx_n=0
idx_mode=0

k=${KS[$idx_k]}
n=${NS[$idx_n]}
mode=${MODES[$idx_mode]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --mode "${mode}" \
  --device "cpu"
