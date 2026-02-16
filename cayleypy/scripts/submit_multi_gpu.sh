#!/bin/bash
# Submit a multi-GPU BFS job on Hyak.
#
# Queries hyakalloc for a partition with enough free GPUs to meet the
# requested total VRAM, then submits the job via sbatch.
#
# Usage:
#   ./cayleypy/scripts/submit_multi_gpu.sh --k 10 --n 35
#   ./cayleypy/scripts/submit_multi_gpu.sh --k 10 --n 35 --min_vram 70
#   ./cayleypy/scripts/submit_multi_gpu.sh --k 10 --n 37 --min_vram 280 --time 24:00:00
#   ./cayleypy/scripts/submit_multi_gpu.sh --k 10 --n 38 --min_vram 520 --time 24:00:00
#   ./cayleypy/scripts/submit_multi_gpu.sh --k 10 --n 35 --account amath --partition gpu-rtx6k

set -euo pipefail

# --- Defaults ---
MIN_VRAM=192          # minimum total GPU VRAM in GB, there is 192Gb in the 8 rtx6k GPUs for the amath account, partition gpu-rtx6k
K=""
N=""
ACCOUNT=""
PARTITION=""
TIME="12:00:00"
GENERATOR_FAMILY="consecutive"
COSET="True"
CENTRAL_MODE="block"
INVERSE_CLOSED="False"
MEM="10G"

# Known per-GPU VRAM (GB) for each partition type.
declare -A GPU_VRAM=(
    [gpu-h200]=141
    [gpu-a100]=80
    [gpu-a40]=48
    [gpu-l40]=48
    [gpu-l40s]=48
    [gpu-rtx6k]=24
    [gpu-2080ti]=11
    [gpu-p100]=16
)

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k)          K="$2"; shift 2 ;;
        --n)          N="$2"; shift 2 ;;
        --min_vram)   MIN_VRAM="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --time)       TIME="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --generator_family) GENERATOR_FAMILY="$2"; shift 2 ;;
        --coset)      COSET="$2"; shift 2 ;;
        --central_mode) CENTRAL_MODE="$2"; shift 2 ;;
        --inverse_closed) INVERSE_CLOSED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$K" || -z "$N" ]]; then
    echo "Error: --k and --n are required."
    echo "Usage: $0 --k 10 --n 35 [--min_vram 200] [--account X] [--partition Y]"
    exit 1
fi

# --- Find a suitable partition ---
if [[ -n "$ACCOUNT" && -n "$PARTITION" ]]; then
    # User specified account/partition — just figure out num_gpus.
    VRAM_PER_GPU=${GPU_VRAM[$PARTITION]:-0}
    if [[ "$VRAM_PER_GPU" -eq 0 ]]; then
        echo "Error: Unknown GPU VRAM for partition '$PARTITION'."
        echo "Known partitions: ${!GPU_VRAM[*]}"
        exit 1
    fi
    NUM_GPUS=$(( (MIN_VRAM + VRAM_PER_GPU - 1) / VRAM_PER_GPU ))

    # Check free GPUs from hyakalloc.
    FREE_GPUS=$(hyakalloc 2>/dev/null | grep -A2 "│ *${ACCOUNT} *│ *${PARTITION} *│" \
        | grep "FREE" | awk -F'│' '{print $5}' | tr -d ' ')
    if [[ -z "$FREE_GPUS" ]]; then
        FREE_GPUS=0
    fi
    if [[ "$NUM_GPUS" -gt "$FREE_GPUS" ]]; then
        echo "Warning: Need $NUM_GPUS GPUs but only $FREE_GPUS free on $ACCOUNT/$PARTITION."
        echo "Job will be queued until resources become available."
    fi
else
    # Auto-detect: parse hyakalloc and find the best partition.
    echo "Searching for a partition with >= ${MIN_VRAM} GB total GPU VRAM..."

    # Preferred accounts in priority order.
    PREFERRED_ACCOUNTS=(amath krishna cse stf)

    # Collect candidates from FREE rows (available now) and TOTAL rows (for queuing).
    declare -a FREE_CANDIDATES=()
    declare -a TOTAL_CANDIDATES=()

    cur_acct=""
    cur_part=""
    while IFS='│' read -r _ acct part cpus mem gpus status _; do
        acct=$(echo "$acct" | xargs)
        part=$(echo "$part" | xargs)
        gpus=$(echo "$gpus" | xargs)
        status=$(echo "$status" | xargs)

        # Carry forward account/partition from the TOTAL row.
        [[ -n "$acct" ]] && cur_acct="$acct"
        [[ -n "$part" ]] && cur_part="$part"

        [[ ! "$cur_part" =~ ^gpu- ]] && continue
        [[ -z "$gpus" || "$gpus" == "0" ]] && continue

        VRAM_PER_GPU=${GPU_VRAM[$cur_part]:-0}
        [[ "$VRAM_PER_GPU" -eq 0 ]] && continue

        num_gpus=$((gpus))
        total_vram=$((num_gpus * VRAM_PER_GPU))

        if [[ "$total_vram" -ge "$MIN_VRAM" ]]; then
            needed=$(( (MIN_VRAM + VRAM_PER_GPU - 1) / VRAM_PER_GPU ))
            if [[ "$status" == "FREE" ]]; then
                FREE_CANDIDATES+=("${cur_acct}|${cur_part}|${needed}|${VRAM_PER_GPU}")
            elif [[ "$status" == "TOTAL" ]]; then
                TOTAL_CANDIDATES+=("${cur_acct}|${cur_part}|${needed}|${VRAM_PER_GPU}")
            fi
        fi
    done < <(hyakalloc 2>/dev/null | grep "│")

    # Helper: pick best from a candidate list by account priority > fewer GPUs > more VRAM/GPU.
    pick_best() {
        local -n cands=$1
        BEST_ACCOUNT=""
        BEST_PARTITION=""
        BEST_NUM_GPUS=999
        BEST_VRAM_PER_GPU=0
        BEST_ACCT_PRIO=999

        for cand in "${cands[@]}"; do
            IFS='|' read -r c_acct c_part c_needed c_vram <<< "$cand"

            c_prio=999
            for i in "${!PREFERRED_ACCOUNTS[@]}"; do
                if [[ "${PREFERRED_ACCOUNTS[$i]}" == "$c_acct" ]]; then
                    c_prio=$i
                    break
                fi
            done

            if [[ "$c_prio" -lt "$BEST_ACCT_PRIO" ]] || \
               [[ "$c_prio" -eq "$BEST_ACCT_PRIO" && "$c_needed" -lt "$BEST_NUM_GPUS" ]] || \
               [[ "$c_prio" -eq "$BEST_ACCT_PRIO" && "$c_needed" -eq "$BEST_NUM_GPUS" && "$c_vram" -gt "$BEST_VRAM_PER_GPU" ]]; then
                BEST_ACCOUNT="$c_acct"
                BEST_PARTITION="$c_part"
                BEST_NUM_GPUS="$c_needed"
                BEST_VRAM_PER_GPU="$c_vram"
                BEST_ACCT_PRIO="$c_prio"
            fi
        done
    }

    # Prefer free GPUs; fall back to TOTAL (job will queue).
    if [[ ${#FREE_CANDIDATES[@]} -gt 0 ]]; then
        pick_best FREE_CANDIDATES
        echo "Found (free now): account=$BEST_ACCOUNT partition=$BEST_PARTITION (${BEST_VRAM_PER_GPU}GB/GPU × ${BEST_NUM_GPUS} = $((BEST_VRAM_PER_GPU * BEST_NUM_GPUS))GB total)"
    elif [[ ${#TOTAL_CANDIDATES[@]} -gt 0 ]]; then
        pick_best TOTAL_CANDIDATES
        echo "No free GPUs available. Will queue on: account=$BEST_ACCOUNT partition=$BEST_PARTITION (${BEST_VRAM_PER_GPU}GB/GPU × ${BEST_NUM_GPUS} = $((BEST_VRAM_PER_GPU * BEST_NUM_GPUS))GB total)"
    else
        echo "Error: No partition found with >= ${MIN_VRAM} GB total GPU VRAM."
        echo "Try a lower --min_vram or specify --account and --partition manually."
        exit 1
    fi

    ACCOUNT="$BEST_ACCOUNT"
    PARTITION="$BEST_PARTITION"
    NUM_GPUS="$BEST_NUM_GPUS"
    VRAM_PER_GPU="$BEST_VRAM_PER_GPU"
fi

echo "Submitting: k=$K n=$N account=$ACCOUNT partition=$PARTITION num_gpus=$NUM_GPUS time=$TIME"

# --- Submit via sbatch ---
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=k${K}_n${N}_${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=2
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=logs/%x_%N_%A.out

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate cayley

srun python3 cayleypy/experiments/consecutive_k_cycles.py \\
  --k ${K} \\
  --n ${N} \\
  --device cuda \\
  --generator_family ${GENERATOR_FAMILY} \\
  --coset ${COSET} \\
  --central_mode ${CENTRAL_MODE} \\
  --inverse_closed ${INVERSE_CLOSED} \\
  --num_gpus ${NUM_GPUS}
EOF
