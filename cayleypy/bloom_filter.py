import math

import torch


class BloomFilter:
    """Bit-packed Bloom filter for memory-efficient seen-state tracking in BFS.

    Uses bit-packed ``torch.uint8`` storage, giving roughly
    ``-log2(false_positive_rate) / log(2)`` bits per element.
    At a 1 % false positive rate that is ~9.6 bits/element vs. 64 bits for
    an ``int64`` hash set — approximately a **6.5x memory reduction**.

    Expects inputs that are already int64 hashes (as produced by the BFS
    hasher).  k bit positions are derived via double hashing — splitting
    the 64-bit value into two 32-bit halves h1/h2 and computing
    ``pos_i = (h1 + i * h2) % num_bits`` for i in 0..k-1.  This avoids
    any additional hash computation; the arithmetic is a single vectorised
    ``(k, n)`` broadcast over the n incoming hashes.

    False positives cause BFS to occasionally skip a state that has not
    actually been visited, which may result in a slightly incomplete
    exploration.  False negatives never occur.

    Args:
        capacity: Expected maximum number of elements to insert.
        false_positive_rate: Target false positive probability (default 1 %).
        device: PyTorch device for the bit array (CPU or CUDA).
        seed: Unused; kept for API compatibility.
    """

    def __init__(
        self,
        capacity: int,
        false_positive_rate: float = 0.01,
        device=None,
        seed: int = 42,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be in (0, 1)")

        # Optimal number of bits (m) and hash functions (k).
        # Round m up to the next power of 2 so that modulo becomes a cheap
        # bitwise AND in _compute_positions.  Memory overhead is at most 2×
        # (typically ~1.3×) vs. the exact optimal size.
        m_exact = max(64, int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2)))
        m = 1 << (m_exact - 1).bit_length()  # next power of 2
        k = max(1, round((m / capacity) * math.log(2)))

        self.num_bits = m
        self.num_bits_mask = m - 1  # fast modulo: x % m == x & mask when m is power of 2
        self.num_hashes = k
        self.device = device
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate

        # Bit-packed storage: num_bits // 8 bytes.
        self.bits = torch.zeros(m // 8, dtype=torch.uint8, device=device)

        # i = 0, 1, ..., k-1 — used in the double-hashing formula.
        self._i = torch.arange(k, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_positions(
        self, hashes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive k byte/bit positions per hash via double hashing.

        BFS hashes may only occupy a fraction of the int64 range (e.g.
        dot-product hashes of permutation graphs sit in ~35 bits, leaving
        the upper 32 bits near-zero).  Splitting such values directly into
        two halves collapses h2 to 0 for many inputs, making all k positions
        identical.

        Fix: one multiply by the splitmix64 constant mixes all bits across
        the full 64-bit range first (cheap, O(n)).  We then split into two
        independent 32-bit halves and apply:
            pos_i = (h1 + i * h2) % num_bits   for i in 0..k-1

        ``num_bits`` is rounded up to a power of 2 so that the modulo
        reduces to a bitwise AND (``& num_bits_mask``), eliminating
        ``torch.remainder`` entirely.

        Returns:
            byte_idx: shape ``(k, n)``, dtype int64
            bit_pos:  shape ``(k, n)``, dtype uint8
        """
        h = hashes * 0xBF58476D1CE4E5B9                  # (n,) mix bits (splitmix64 step 1)
        h1 = h & 0xFFFFFFFF                              # (n,) lower 32 bits
        h2 = ((h >> 32) & 0xFFFFFFFF) | 1               # (n,) upper 32 bits, ensure nonzero
        i = self._i.to(hashes.device)[:, None]           # (k, 1)
        raw = h1[None, :] + i * h2[None, :]              # (k, n)
        pos = raw & self.num_bits_mask                    # (k, n) fast modulo — valid because num_bits is a power of 2
        return (pos >> 3).to(torch.int64), (pos & 7).to(torch.uint8)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, hashes: torch.Tensor) -> None:
        """Mark all *hashes* as seen in the filter."""
        if len(hashes) == 0:
            return
        hashes = hashes.to(self.bits.device)
        byte_idx, bit_pos = self._compute_positions(hashes)  # (k, n) each
        byte_idx_flat = byte_idx.view(-1)                     # (k*n,)
        bit_pos_flat = bit_pos.view(-1)                       # (k*n,)
        for b in range(8):
            mask = bit_pos_flat == b
            if mask.any():
                self.bits[byte_idx_flat[mask]] |= (1 << b)

    def contains(self, hashes: torch.Tensor) -> torch.Tensor:
        """Return a bool tensor: ``True`` where a hash was *probably* added.

        False positives are possible; false negatives are not.
        """
        if len(hashes) == 0:
            return torch.zeros(0, dtype=torch.bool, device=hashes.device)

        byte_idx, bit_pos = self._compute_positions(hashes)    # (k, n)
        byte_vals = self.bits.to(hashes.device)[byte_idx]      # (k, n)
        return ((byte_vals >> bit_pos) & 1).bool().all(dim=0)  # (n,)

    def memory_bytes(self) -> int:
        """Return the number of bytes used by the bit array."""
        return self.bits.nbytes

    def __repr__(self) -> str:
        return (
            f"BloomFilter(capacity={self.capacity}, "
            f"false_positive_rate={self.false_positive_rate}, "
            f"num_bits={self.num_bits}, num_hashes={self.num_hashes}, "
            f"memory_bytes={self.memory_bytes()})"
        )
