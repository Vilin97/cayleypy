#!/bin/bash
#SBATCH --job-name=cycles_k2_6_n13_14
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%N_%A_%a.out

# ------------------------
# PARAMETER GRIDS
# ------------------------
GENERATOR_FAMILIES=(consecutive wrapped)
KS=(2 3 4 5 6)
NS=(13 14)
COINCIDES=(0 2 3 4 5 6)

DEVICE=cuda

# ------------------------
# INDEX UNPACKING
# ------------------------
tid=$SLURM_ARRAY_TASK_ID

n_coincide_idx=$(( tid % ${#COINCIDES[@]} ))
tid=$(( tid / ${#COINCIDES[@]} ))

n_idx=$(( tid % ${#NS[@]} ))
tid=$(( tid / ${#NS[@]} ))

k_idx=$(( tid % ${#KS[@]} ))
tid=$(( tid / ${#KS[@]} ))

gen_idx=$(( tid % ${#GENERATOR_FAMILIES[@]} ))

GENERATOR_FAMILY=${GENERATOR_FAMILIES[$gen_idx]}
K=${KS[$k_idx]}
N=${NS[$n_idx]}
N_COINCIDE=${COINCIDES[$n_coincide_idx]}

echo "Running:"
echo "  generator_family = ${GENERATOR_FAMILY}"
echo "  k                = ${K}"
echo "  n                = ${N}"
echo "  n_coincide        = ${N_COINCIDE}"
echo "  inverse_closed    = False"

# ------------------------
# ENVIRONMENT
# ------------------------
source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

# ------------------------
# RUN
# ------------------------
srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${K}" \
  --n "${N}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --n_coincide "${N_COINCIDE}"
