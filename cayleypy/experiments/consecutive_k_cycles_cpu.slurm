#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=15
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(7)
NS=(37)
MODES=(consecutive wrapped)

tid=$SLURM_ARRAY_TASK_ID

idx_mode=$(( tid % 2 ))
idx_k=0
idx_n=0

mode=${MODES[$idx_mode]}
k=${KS[$idx_k]}
n=${NS[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --mode "${mode}" \
  --device "cpu"
