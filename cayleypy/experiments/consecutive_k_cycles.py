"python cayleypy/experiments/consecutive_k_cycles.py --k 7 --n 25"

import argparse
import time
import torch
import wandb
from cayleypy import CayleyGraph, PermutationGroups

def run_single_n(k: int, n: int):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    central = [0] * (n // 2) + [1] * (n - n // 2)
    defn = PermutationGroups.consecutive_k_cycles(n, k).with_central_state(central)
    graph = CayleyGraph(defn)
    result = graph.bfs(return_all_edges=False, return_all_hashes=False)

    runtime = time.time() - t0
    diameter = result.diameter()
    layer_sizes = result.layer_sizes

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()

    print(f"n={n}, diameter: {diameter}, layer sizes: {layer_sizes}")
    print(f"Peak GPU memory: {peak_bytes / 1024**3:.3f} GiB")
    print(f"Runtime: {runtime:.3f} seconds")

    wandb.init(
        project="cayley_consecutive_k_cycles",
        name=f"k_{k}_n_{n}",
        config={"k": k, "n": n},
    )

    wandb.log(
        {
            "diameter": diameter,
            "peak_memory_bytes": peak_bytes,
            "peak_memory_gib": peak_bytes / 1024**3,
            "num_layers": len(layer_sizes),
            "layer_sizes": layer_sizes,
            "runtime_sec": runtime,
        }
    )
    wandb.finish()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_single_n(args.k, args.n)
