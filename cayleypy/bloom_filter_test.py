import torch
import pytest

from .bloom_filter import BloomFilter
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


# ---------------------------------------------------------------------------
# BloomFilter unit tests
# ---------------------------------------------------------------------------

def test_no_false_negatives():
    """Every added hash must be found — false negatives must never occur."""
    bf = BloomFilter(capacity=10_000)
    hashes = torch.randint(-2**62, 2**62, (5_000,), dtype=torch.int64)
    bf.add(hashes)
    assert bf.contains(hashes).all(), "False negative detected"


def test_empty_add_and_contains():
    """Empty tensors should not raise and should return empty results."""
    bf = BloomFilter(capacity=1_000)
    bf.add(torch.empty(0, dtype=torch.int64))
    result = bf.contains(torch.empty(0, dtype=torch.int64))
    assert result.shape == (0,)


def test_false_positive_rate_within_bounds():
    """Measured FPR on unseen hashes should be reasonably close to the target."""
    target_fpr = 0.01
    bf = BloomFilter(capacity=100_000, false_positive_rate=target_fpr)
    seen = torch.randint(-2**62, 2**62, (100_000,), dtype=torch.int64)
    bf.add(seen)

    unseen = torch.randint(-2**62, 2**62, (200_000,), dtype=torch.int64)
    measured_fpr = bf.contains(unseen).float().mean().item()
    # Allow 3× margin — Bloom filters are probabilistic
    assert measured_fpr < target_fpr * 3, f"FPR too high: {measured_fpr:.4f}"


def test_memory_smaller_than_hash_set():
    """Bit-packed filter should use less memory than storing int64 hashes."""
    capacity = 1_000_000
    bf = BloomFilter(capacity=capacity, false_positive_rate=0.01)
    hash_set_bytes = capacity * 8  # 8 bytes per int64
    assert bf.memory_bytes() < hash_set_bytes, (
        f"Filter ({bf.memory_bytes()} B) should be smaller than hash set ({hash_set_bytes} B)"
    )


def test_incremental_add():
    """Adding hashes in batches should give the same result as adding all at once."""
    hashes = torch.randint(-2**62, 2**62, (10_000,), dtype=torch.int64)

    bf_all = BloomFilter(capacity=10_000, seed=1)
    bf_all.add(hashes)

    bf_incremental = BloomFilter(capacity=10_000, seed=1)
    for chunk in hashes.split(1_000):
        bf_incremental.add(chunk)

    assert torch.equal(bf_all.contains(hashes), bf_incremental.contains(hashes))


# ---------------------------------------------------------------------------
# BFS correctness tests: bloom filter must match exact BFS layer sizes
# ---------------------------------------------------------------------------

def _bfs_layer_sizes(graph, **kwargs):
    return graph.bfs(**kwargs).layer_sizes


def test_bfs_bloom_matches_exact_small():
    """Bloom filter BFS must produce identical layer sizes on a small graph."""
    graph = CayleyGraph(PermutationGroups.lrx(5).with_central_state("01210"))

    exact = _bfs_layer_sizes(graph)
    bloom = _bfs_layer_sizes(graph, bloom_filter_capacity=10_000)

    assert exact == bloom, f"Mismatch:\n  exact={exact}\n  bloom={bloom}"


def test_bfs_bloom_matches_exact_swap():
    """Bloom filter BFS on a trivial 2-state graph."""
    from .cayley_graph_def import CayleyGraphDef
    graph = CayleyGraph(CayleyGraphDef.create([[1, 0]], central_state="01"))

    exact = _bfs_layer_sizes(graph)
    bloom = _bfs_layer_sizes(graph, bloom_filter_capacity=1_000)

    assert exact == bloom


def test_bfs_bloom_matches_exact_directed():
    """Bloom filter should work correctly on a non-inverse-closed (directed) graph."""
    graph = CayleyGraph(PermutationGroups.lrx(6))

    exact = _bfs_layer_sizes(graph)
    bloom = _bfs_layer_sizes(graph, bloom_filter_capacity=50_000)

    assert exact == bloom, f"Mismatch:\n  exact={exact}\n  bloom={bloom}"


def test_bfs_bloom_completed():
    """BFS with bloom filter should still report completion correctly."""
    graph = CayleyGraph(PermutationGroups.lrx(5).with_central_state("01210"))
    result = graph.bfs(bloom_filter_capacity=10_000)
    assert result.bfs_completed
