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
    coset: bool,
    central_mode: str,
    inverse_closed: bool,
):
    wandb.init(
        entity="CayleyPy",
        project="cycles",
        name=(
            f"k_{k}_n_{n}_{generator_family}_{device}_"
            f"coset_{int(coset)}_{central_mode if coset else 'full'}_"
            f"inv_{int(inverse_closed)}"
        ),
        config={
            "k": k,
            "n": n,
            "generator_family": generator_family,
            "device": device,
            "coset": coset,
            "central_mode": central_mode,
            "inverse_closed": inverse_closed,
        },
    )

    print(
        f"Running k={k} n={n} gen={generator_family} device={device} "
        f"coset={coset} central_mode={central_mode} inverse_closed={inverse_closed}"
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

    # IMPORTANT: full graph vs coset
    if coset:
        if central_mode == "alternating":
            central = [0, 1] * (n // 2) + [0] * (n - 2 * (n // 2))
        elif central_mode == "block":
            central = [0] * (n // 2) + [1] * (n - n // 2)
        else:
            raise ValueError(f"Invalid central_mode: {central_mode}")
        defn = defn.with_central_state(central)

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
            coset=coset,
            central_mode=central_mode if coset else None,
            inverse_closed=inverse_closed,
        )
    )
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, help="cycle length for the generators")
    p.add_argument("--n", type=int, required=True, help="number of elements being permuted (group is S_n or a coset thereof)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="compute device for BFS (default: cuda)")
    p.add_argument("--generator_family", choices=["consecutive", "wrapped"], default="consecutive", help="'consecutive' uses k-cycles on adjacent elements (1..k, 2..k+1, ...); 'wrapped' wraps around so position n connects to position 1 (default: consecutive)")
    p.add_argument("--coset", type={"True": True, "False": False}.__getitem__, default=True, help="if True, restrict to a coset via a central state coloring, reducing the graph from n! to C(n, n/2) vertices; if False, BFS the full Cayley graph (default: True)")
    p.add_argument("--central_mode", choices=["alternating", "block"], default="block", help="how to color positions for the coset: 'block' puts first half as 0 and second half as 1; 'alternating' interleaves 0,1,0,1,... Only used when --coset True (default: block)")
    p.add_argument("--inverse_closed", type={"True": True, "False": False}.__getitem__, default=False, help="if True, add the inverse of each generator to the generating set, making the Cayley graph undirected (default: False)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_single_n(
        a.k,
        a.n,
        a.generator_family,
        a.device,
        a.coset,
        a.central_mode,
        a.inverse_closed,
    )