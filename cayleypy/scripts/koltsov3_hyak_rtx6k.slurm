#!/bin/bash
#SBATCH --job-name=koltsov3_small
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-883
#SBATCH --output=logs/%x_%N_%A_%a.out

# n=5..30 on RTX 6000 (24GB VRAM) — coset sizes fit comfortably
LS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34)
NS=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

GENERATOR_FAMILY=koltsov3
DEVICE=cuda
K=1
PERM_TYPE=2

tid=$SLURM_ARRAY_TASK_ID
num_n=${#NS[@]}
l_idx=$((tid / num_n))
n_idx=$((tid % num_n))
L=${LS[$l_idx]}
n=${NS[$n_idx]}

# skip invalid combinations: need L < n
if [ "$L" -ge "$n" ]; then
  echo "Skipping L=$L n=$n (L >= n)"
  exit 0
fi

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${K}" \
  --n "${n}" \
  --L "${L}" \
  --device "${DEVICE}" \
  --generator_family "${GENERATOR_FAMILY}" \
  --coset True \
  --central_mode L \
  --inverse_closed False \
  --perm_type "${PERM_TYPE}"
