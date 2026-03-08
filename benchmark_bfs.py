"""
BFS benchmark: bloom filter vs baseline.
Shows speed, memory, layer structure, and cProfile breakdown of a real BFS run.

Memory is measured via /proc/self/status (VmRSS) which captures torch tensor
allocations that tracemalloc misses.
"""

import cProfile
import io
import os
import pstats
import time

import torch

from cayleypy import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Resident set size in MB from /proc/self/status."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024  # kB -> MB
    return float("nan")


def _bloom_memory_mb(capacity: int, fpr: float = 0.01) -> float:
    """Report the bloom filter's own bit-array size in MB."""
    from cayleypy.bloom_filter import BloomFilter
    bf = BloomFilter(capacity, fpr)
    return bf.memory_bytes() / (1024 ** 2)


def _hash_set_memory_mb(n_states: int) -> float:
    """Bytes to store n_states as int64 hashes."""
    return n_states * 8 / (1024 ** 2)


# ---------------------------------------------------------------------------
# BFS runner
# ---------------------------------------------------------------------------

def run_bfs(graph, bloom_capacity=None, label=""):
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    result = graph.bfs(bloom_filter_capacity=bloom_capacity)
    elapsed = time.perf_counter() - t0
    rss_after = _rss_mb()
    delta_mb = rss_after - rss_before
    return result, elapsed, delta_mb


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_layer_table(result, label):
    print(f"\n  Layer structure — {label}")
    print(f"  {'Layer':>6}  {'New states':>12}  {'Cumulative':>12}")
    cumulative = 0
    for i, size in enumerate(result.layer_sizes):
        cumulative += size
        print(f"  {i:>6}  {size:>12,}  {cumulative:>12,}")
    print(f"  Diameter: {len(result.layer_sizes)-1}   Total states: {cumulative:,}")


def print_comparison(t_exact, mem_exact, t_bloom, mem_bloom,
                     total_states, bloom_capacity, correct):
    hash_set_mb = _hash_set_memory_mb(total_states)
    bloom_mb = _bloom_memory_mb(bloom_capacity)
    speedup = t_exact / t_bloom if t_bloom > 0 else float("inf")

    print(f"\n  {'Metric':<35} {'Exact':>10}  {'Bloom':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Wall time (s)':<35} {t_exact:>10.3f}  {t_bloom:>10.3f}  "
          f"({'bloom {:.1f}x faster'.format(speedup) if speedup > 1 else 'bloom {:.1f}x slower'.format(1/speedup)})")
    print(f"  {'RSS delta (MB)':<35} {mem_exact:>10.1f}  {mem_bloom:>10.1f}")
    print(f"  {'Seen-set memory (MB, estimated)':<35} {hash_set_mb:>10.1f}  {bloom_mb:>10.1f}  "
          f"({hash_set_mb/bloom_mb:.1f}x smaller)")
    print(f"  {'Layer sizes match':<35} {'':>10}  {'YES' if correct else 'NO':>10}")


# ---------------------------------------------------------------------------
# Full cProfile of one BFS run
# ---------------------------------------------------------------------------

def profile_bfs(graph_def, graph_label, bloom_capacity):
    print(f"\n{'='*65}")
    print(f"  cProfile — {graph_label}")
    print(f"{'='*65}")

    for mode, kw in [("bloom filter", dict(bloom_filter_capacity=bloom_capacity)),
                     ("exact baseline", {})]:
        graph = CayleyGraph(graph_def, device="cpu")
        pr = cProfile.Profile()
        pr.enable()
        graph.bfs(**kw)
        pr.disable()

        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
        ps.print_stats(18)
        raw = buf.getvalue()
        # Strip long install paths for readability
        for prefix in [".venv/lib/python3.12/site-packages/", "/home/"]:
            raw = raw.replace(os.path.dirname(__file__) + "/", "")
        print(f"\n  --- {mode} ---")
        print(raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GRAPHS = [
        ("all_transpositions(8)  [40k states]",
         PermutationGroups.all_transpositions(8), 100_000),
        ("all_transpositions(9)  [362k states]",
         PermutationGroups.all_transpositions(9), 500_000),
    ]

    for label, graph_def, bloom_cap in GRAPHS:
        print(f"\n{'='*65}")
        print(f"  Graph: {label}")
        print(f"{'='*65}")

        graph = CayleyGraph(graph_def, device="cpu")

        result_exact, t_exact, mem_exact = run_bfs(graph, label="exact")
        print_layer_table(result_exact, "exact (hash-set baseline)")

        result_bloom, t_bloom, mem_bloom = run_bfs(graph, bloom_capacity=bloom_cap, label="bloom")
        print_layer_table(result_bloom, f"bloom filter (capacity={bloom_cap:,})")

        total = sum(result_exact.layer_sizes)
        correct = result_exact.layer_sizes == result_bloom.layer_sizes
        print_comparison(t_exact, mem_exact, t_bloom, mem_bloom, total, bloom_cap, correct)

    # Full cProfile on the larger graph
    profile_bfs(
        PermutationGroups.all_transpositions(9),
        "all_transpositions(9)  [362k states]",
        bloom_capacity=500_000,
    )
