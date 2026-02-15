"""Benchmark: single-GPU vs multi-GPU BFS with distributed hash-partitioned BFS.

Measures peak GPU memory (per device) and wall-clock time for consecutive k-cycle
coset graphs with k=10, varying n. Compares num_gpus=1 vs num_gpus=N.

Usage:
    python bench_multi_gpu.py                              # defaults: n=25..32, 1 vs all GPUs
    python bench_multi_gpu.py --n_values 28 29 30          # specific n values
    python bench_multi_gpu.py --gpu_configs 1 4 7          # specific GPU counts

Results (7x Quadro RTX 6000, k=10, consecutive k-cycles coset w/ block central):

Single-GPU baseline vs 7-GPU distributed:

   n  gpus      vertices  diam   GPU0_GiB  total_GiB   time_s  GPU0_saved
  25     1     5,200,300    24      0.322      0.322     0.30       -
  25     7     5,200,300    24      0.305      0.677     0.40     +5.1%
  27     1    20,058,300    26      0.675      0.675     0.52       -
  27     7    20,058,300    26      0.632      1.351     0.70     +6.3%
  29     1    77,558,760    29      1.199      1.199     2.54       -
  29     7    77,558,760    29      0.857      2.122     3.15    +28.5%
  30     1   155,117,520    31      2.015      2.015     6.13       -
  30     7   155,117,520    31      0.534      3.290     7.24    +73.5%
  31     1   300,540,195    33      3.755      3.755    15.03       -
  31     7   300,540,195    33      0.618      5.618    17.10    +83.5%
  32     1   601,080,390    35      7.202      7.202    41.51       -
  32     7   601,080,390    35      1.039     10.273    44.86    +85.6%

7-GPU only (single-GPU would OOM):

   n  gpus       vertices  diam   GPU0_GiB  max_GPU_GiB   time_s
  33     7  1,166,803,110    37      1.963        1.963    153.1s
  34     7  2,333,606,220    39      3.708        3.708    332.0s
  35     7  4,537,567,650    41      7.304        7.304    737.8s
  36     7  9,075,135,300    43     14.075       14.075   2026.4s
  37     7           OOM     --       OOM          OOM       OOM
"""
import argparse
import gc
import time

import torch
from cayleypy import CayleyGraph, PermutationGroups


def run_one(k: int, n: int, num_gpus: int) -> dict:
    """Run a single BFS and return measurements."""
    defn = PermutationGroups.consecutive_k_cycles(n, k)
    central = [0] * (n // 2) + [1] * (n - n // 2)
    defn = defn.with_central_state(central)

    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.time()
    graph = CayleyGraph(defn, device="cuda", num_gpus=num_gpus, verbose=0)
    result = graph.bfs()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Collect peak memory for each GPU.
    peak_per_gpu = {}
    for i in range(min(num_gpus, torch.cuda.device_count())):
        peak_per_gpu[i] = torch.cuda.max_memory_allocated(i)

    return {
        "k": k,
        "n": n,
        "num_gpus": num_gpus,
        "diameter": result.diameter(),
        "num_vertices": sum(result.layer_sizes),
        "peak_gpu0_gib": peak_per_gpu.get(0, 0) / (1024**3),
        "peak_total_gib": sum(peak_per_gpu.values()) / (1024**3),
        "peak_per_gpu": peak_per_gpu,
        "elapsed_sec": elapsed,
        "layer_sizes": result.layer_sizes,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU BFS benchmark")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n_values", type=int, nargs="+", default=[25, 27, 29, 30, 31, 32])
    parser.add_argument("--gpu_configs", type=int, nargs="+", default=None)
    args = parser.parse_args()

    n_available = torch.cuda.device_count()
    if args.gpu_configs is None:
        args.gpu_configs = [1, n_available]
    # Remove duplicates and sort.
    args.gpu_configs = sorted(set(g for g in args.gpu_configs if g <= n_available))

    print(f"Available GPUs: {n_available}")
    for i in range(n_available):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"k={args.k}, n_values={args.n_values}, gpu_configs={args.gpu_configs}")
    print()

    results = []
    for n in args.n_values:
        for ng in args.gpu_configs:
            label = f"k={args.k} n={n} num_gpus={ng}"
            print(f"--- {label} ---")
            try:
                r = run_one(args.k, n, ng)
                results.append(r)
                print(f"  vertices={r['num_vertices']:,}  diameter={r['diameter']}")
                print(f"  peak GPU0={r['peak_gpu0_gib']:.3f} GiB  total={r['peak_total_gib']:.3f} GiB  time={r['elapsed_sec']:.2f}s")
                per_gpu_str = ", ".join(f"GPU{i}={v/(1024**3):.3f}" for i, v in sorted(r['peak_per_gpu'].items()))
                print(f"  per-GPU: {per_gpu_str}")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM!")
                results.append({"k": args.k, "n": n, "num_gpus": ng, "oom": True})
                gc.collect()
                for i in range(n_available):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            print()

    # Summary table.
    print("=" * 100)
    print(f"{'n':>4} {'gpus':>5} {'vertices':>12} {'GPU0_GiB':>10} {'total_GiB':>10} {'time_s':>8} {'GPU0_saved':>10} {'speedup':>8}")
    print("-" * 100)
    for r in results:
        if r.get("oom"):
            print(f"{r['n']:>4} {r['num_gpus']:>5} {'OOM':>12}")
            continue
        baseline = [
            x for x in results
            if x.get("n") == r["n"] and x.get("num_gpus") == 1 and not x.get("oom")
        ]
        if baseline:
            gpu0_saved = (1 - r["peak_gpu0_gib"] / baseline[0]["peak_gpu0_gib"]) * 100
            gpu0_str = f"{gpu0_saved:+.1f}%"
            speedup = baseline[0]["elapsed_sec"] / r["elapsed_sec"]
            speedup_str = f"{speedup:.2f}x"
        else:
            gpu0_str = "N/A"
            speedup_str = "N/A"
        print(
            f"{r['n']:>4} {r['num_gpus']:>5} {r['num_vertices']:>12,} "
            f"{r['peak_gpu0_gib']:>10.3f} {r['peak_total_gib']:>10.3f} "
            f"{r['elapsed_sec']:>8.2f} {gpu0_str:>10} {speedup_str:>8}"
        )


if __name__ == "__main__":
    main()
