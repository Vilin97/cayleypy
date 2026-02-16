#!/bin/bash
# Submit k=10 n=36,37 multi-GPU BFS jobs.
#
# n=36: ~14 GiB/GPU on 7x RTX 6000, 9.1B vertices, ~34 min
# n=37: OOM'd on 7x RTX 6000 (~21 GiB/GPU), needs bigger GPUs
#       estimated ~200 GiB total data + sort overhead → 300 GiB VRAM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# n=36: 200 GiB total is enough (e.g. 5x L40S or 3x A100)
"$SCRIPT_DIR/submit_multi_gpu.sh" --k 10 --n 36 --min_vram 192 --time 4:00:00

# n=37: needs ~300 GiB to handle sort transients (e.g. 4x A100 or 3x H200)
"$SCRIPT_DIR/submit_multi_gpu.sh" --k 10 --n 37 --min_vram 300 --time 8:00:00
