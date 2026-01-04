import argparse
import time
import psutil
import torch
import wandb
import platform
from cayleypy import CayleyGraph, PermutationGroups


def run_single_n(
    k: int,
    n: int,
    generator_family: str,
    device: str,
    inverse_closed: bool,
    n_coincide: int,
):
    # ---- PARAMETER VALIDATION ----
    if n_coincide == n:
        raise ValueError("n_coincide == n collapses the graph to a single state; this case is not supported")

    if not (n_coincide == 0 or 2 <= n_coincide <= n - 1):
        raise ValueError("n_coincide must be 0 or satisfy 2 <= n_coincide <= n-1")

    wandb.init(
        entity="CayleyPy",
        project="cycles",
        name=(f"k_{k}_n_{n}_{generator_family}_{device}_" f"coinc_{n_coincide}_inv_{int(inverse_closed)}"),
        config={
            "k": k,
            "n": n,
            "generator_family": generator_family,
            "device": device,
            "inverse_closed": inverse_closed,
            "n_coincide": n_coincide,
        },
    )

    print(
        f"Running k={k} n={n} gen={generator_family} device={device} "
        f"inverse_closed={inverse_closed} n_coincide={n_coincide}"
    )

    proc = psutil.Process()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        torch.cuda.reset_peak_memory_stats()
    mem_before = proc.memory_info().rss

    if device == "cuda":
        print(f"Hardware: GPU = {torch.cuda.get_device_name(0)}")
    else:
        print(f"Hardware: CPU = {platform.processor()}, cores = {psutil.cpu_count(logical=True)}")

    t0 = time.time()

    if generator_family == "consecutive":
        defn = PermutationGroups.consecutive_k_cycles(n, k)
    elif generator_family == "wrapped":
        defn = PermutationGroups.wrapped_k_cycles(n, k)
    else:
        raise ValueError(f"Invalid generator_family: {generator_family}")

    if inverse_closed:
        defn = defn.make_inverse_closed()

    # IMPORTANT: full graph vs coset (implemented via coincide)
    if n_coincide == 1:
        raise ValueError("n_coincide = 1 is invalid")

    if n_coincide > n:
        raise ValueError("n_coincide cannot exceed n")

    if n_coincide >= 2:
        central = list(range(n - n_coincide)) + [n - n_coincide] * n_coincide
        defn = defn.with_central_state(central)
    # else: n_coincide == 0 → full graph

    graph = CayleyGraph(defn)
    result = graph.bfs(return_all_edges=False, return_all_hashes=False)

    last_layer = result.last_layer()
    diameter = result.diameter()
    layer_sizes = result.layer_sizes
    runtime = time.time() - t0

    last_layer_list = [list(row) for row in last_layer]
    last_layer_str = "\n".join(" ".join(map(str, row)) for row in last_layer_list)

    if device == "cuda":
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_after = proc.memory_info().rss
        peak_bytes = max(mem_before, mem_after)

    print(f"n={n}, diameter: {diameter}, layer sizes: {layer_sizes}")
    print(f"Last layer:\n{last_layer_str}")
    print(f"Peak memory: {peak_bytes / 1024**3:.3f} GiB")
    print(f"Runtime: {runtime:.3f} seconds")

    wandb.log(
        dict(
            diameter=diameter,
            num_layers=len(layer_sizes),
            layer_sizes=layer_sizes,
            runtime_sec=runtime,
            peak_memory_bytes=peak_bytes,
            peak_memory_gib=peak_bytes / 1024**3,
            last_layer_str=last_layer_str,
            last_layer_list=last_layer_list,
            k=k,
            n=n,
            generator_family=generator_family,
            device=device,
            n_coincide=n_coincide,
            inverse_closed=inverse_closed,
        )
    )
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--generator_family", choices=["consecutive", "wrapped"], default="consecutive")
    p.add_argument("--inverse_closed", action="store_true")
    p.add_argument("--n_coincide", type=int, default=0, help="0 = full graph; >=2 = coincide coset")
    args = p.parse_args()
    if args.n_coincide == 1:
        raise ValueError("n_coincide=1 is invalid")
    return args


if __name__ == "__main__":
    a = parse_args()
    print("ARGS RAW:", a)
    print("ARGS VALUES:", a.k, a.n, a.n_coincide)

    run_single_n(
        a.k,
        a.n,
        a.generator_family,
        a.device,
        a.inverse_closed,
        a.n_coincide,
    )
