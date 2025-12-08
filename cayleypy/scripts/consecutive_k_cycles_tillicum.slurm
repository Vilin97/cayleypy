#!/bin/bash
#SBATCH --job-name=consecutive_k_cycles
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%N_%A_%a.out

KS=(2 3)
NS=(32 33 34 35 36)
GEN_FAMILIES=(consecutive wrapped)

tid=$SLURM_ARRAY_TASK_ID

idx_k=$(( tid / 10 ))          # 2 ks
rem1=$(( tid % 10 ))
idx_n=$(( rem1 / 2 ))          # 5 ns
idx_g=$(( rem1 % 2 ))          # 2 families

k=${KS[$idx_k]}
n=${NS[$idx_n]}
generator_family=${GEN_FAMILIES[$idx_g]}

source /gpfs/projects/mathai/vilin/miniconda3/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \
  --k "${k}" \
  --n "${n}" \
  --generator_family "${generator_family}" \
  --central_mode alternating \
  --device cuda
