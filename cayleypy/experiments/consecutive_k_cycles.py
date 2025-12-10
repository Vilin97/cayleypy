# "python cayleypy/experiments/consecutive_k_cycles.py --k 7 --n 25 --device cpu --generator_family consecutive --central_mode alternating"

import argparse
import time
import psutil
import torch
import wandb
import platform
from cayleypy import CayleyGraph, PermutationGroups


def run_single_n(k: int, n: int, generator_family: str, device: str, central_mode: str):
    wandb.init(
        entity="CayleyPy"
        project="cycles",
        name=f"k_{k}_n_{n}_{generator_family}_{central_mode}_{device}",
        config={
            "k": k,
            "n": n,
            "generator_family": generator_family,
            "device": device,
            "central_mode": central_mode,
        },
    )
    print(
        f"Running for k={k}, n={n}, generator_family={generator_family}, "
        f"device={device}, central_mode={central_mode}"
    )

    peak_bytes = None
    proc = psutil.Process()

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.reset_peak_memory_stats()
    mem_before = proc.memory_info().rss

    # --- hardware info ---
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware: GPU = {gpu_name}")
    else:
        cpu_name = platform.processor()
        cpu_cores = psutil.cpu_count(logical=True)
        print(f"Hardware: CPU = {cpu_name}, cores = {cpu_cores}")
    # ---------------------

    t0 = time.time()
    if central_mode == "alternating":
        central = [0, 1] * (n // 2) + [0] * (n - 2 * (n // 2))
    elif central_mode == "block":
        central = [0] * (n // 2) + [1] * (n - n // 2)
    else:
        raise ValueError(f"Invalid central_mode: {central_mode}")

    if generator_family == "consecutive":
        defn = PermutationGroups.consecutive_k_cycles(n, k).with_central_state(central)
    elif generator_family == "wrapped_inv":
        defn = (
            PermutationGroups.wrapped_k_cycles(n, k)
            .make_inverse_closed()
            .with_central_state(central)
        )
    else:
        raise ValueError(f"Invalid generator_family: {generator_family}")

    graph = CayleyGraph(defn)
    result = graph.bfs(return_all_edges=False, return_all_hashes=False)

    last_layer = result.last_layer()
    runtime = time.time() - t0
    diameter = result.diameter()
    layer_sizes = result.layer_sizes

    # Convert last_layer to plain Python + nice string for printing/logging
    last_layer_list = [list(row) for row in last_layer]
    last_layer_str = "\n".join(" ".join(map(str, row)) for row in last_layer_list)

    if device == "cuda":
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_after = proc.memory_info().rss
        peak_bytes = max(mem_before, mem_after)

    print(
        f"Running for k={k}, n={n}, generator_family={generator_family}, "
        f"device={device}, central_mode={central_mode}"
    )
    print(f"n={n}, diameter: {diameter}, layer sizes: {layer_sizes}")
    print(f"Last layer:\n{last_layer_str}")
    print(f"Peak memory: {peak_bytes / 1024**3:.3f} GiB")
    print(f"Runtime: {runtime:.3f} seconds")

    wandb.log(
        {
            "diameter": diameter,
            "peak_memory_bytes": peak_bytes,
            "peak_memory_gib": peak_bytes / 1024**3,
            "num_layers": len(layer_sizes),
            "layer_sizes": layer_sizes,
            "runtime_sec": runtime,
            # log as text to avoid histogram
            "last_layer_str": last_layer_str,
            "last_layer_list": last_layer_list,
            "k": k,
            "n": n,
            "generator_family": generator_family,
            "device": device,
            "central_mode": central_mode,
        }
    )
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--generator_family",
        type=str,
        choices=["consecutive", "wrapped"],
        default="consecutive",
    )
    p.add_argument(
        "--central_mode",
        type=str,
        choices=["alternating", "block"],
        default="block",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_n(
        args.k,
        args.n,
        args.generator_family,
        args.device,
        args.central_mode,
    )
