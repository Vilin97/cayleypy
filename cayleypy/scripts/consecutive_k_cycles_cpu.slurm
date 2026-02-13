#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles_cpu
#SBATCH --account=stf
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --cpus-per-task=6
#SBATCH --mem=350G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%N_%A_%a.out

K=10
NS=(34 35 36 37 38)
INVERSE_CLOSED_OPTS=(True False)

GENERATOR_FAMILY=consecutive
DEVICE=cpu

tid=$SLURM_ARRAY_TASK_ID
num_n=${#NS[@]}
ic_idx=$((tid / num_n))
n_idx=$((tid % num_n))
n=${NS[$n_idx]}
inverse_closed=${INVERSE_CLOSED_OPTS[$ic_idx]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${K}" \
  --n "${n}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --coset True \
  --central_mode block \
  --inverse_closed "${inverse_closed}"
