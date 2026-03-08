from typing import Optional

import torch

from ..bloom_filter import BloomFilter
from ..cayley_graph import CayleyGraph
from ..cayley_graph_def import AnyStateType
from ..torch_utils import isin_via_searchsorted


class InteractiveBfs:
    """Interactive breadth-first search that computes layers one by one."""

    def __init__(
        self,
        graph: CayleyGraph,
        start_states: AnyStateType,
        bloom_filter_capacity: Optional[int] = None,
        bloom_filter_false_positive_rate: float = 0.01,
    ):
        self.graph = graph
        self.cur_layer = graph.encode_states(start_states)
        initial_hashes = graph.hasher.make_hashes(self.cur_layer)

        if bloom_filter_capacity is not None:
            self._bloom_filter: Optional[BloomFilter] = BloomFilter(
                bloom_filter_capacity,
                bloom_filter_false_positive_rate,
                device=graph.device,
            )
            self._bloom_filter.add(initial_hashes)
            # Keep only the last two layers as actual sorted tensors so that
            # _remove_seen_states and find_on_last_layer still work correctly.
            self.hashes = [initial_hashes]
        else:
            self._bloom_filter = None
            self.hashes = [initial_hashes]

    def _remove_seen_states(self, hashes: torch.Tensor) -> torch.Tensor:
        """Returns mask where 0s are at positions in `current_layer_hashes` that were seen previously."""
        if self._bloom_filter is not None:
            return ~self._bloom_filter.contains(hashes)
        ans = ~isin_via_searchsorted(hashes, self.hashes[-1])
        if len(self.hashes) > 1:
            ans &= ~isin_via_searchsorted(hashes, self.hashes[-2])
        if not self.graph.definition.generators_inverse_closed:
            for h in self.hashes[:-2]:
                ans &= ~isin_via_searchsorted(hashes, h)
        return ans

    def step(self):
        """Computes next layer of BFS."""
        next_layer, next_layer_hashes = self.graph.get_unique_states(self.graph.get_neighbors(self.cur_layer))
        mask = self._remove_seen_states(next_layer_hashes)
        self.cur_layer = next_layer[mask]
        new_hashes = next_layer_hashes[mask]
        if self._bloom_filter is not None:
            self._bloom_filter.add(new_hashes)
            # Keep only the latest layer for find_on_last_layer.
            self.hashes = [new_hashes]
        else:
            self.hashes.append(new_hashes)

    def find_on_last_layer(self, hashes: torch.Tensor) -> Optional[torch.Tensor]:
        """Checks if ``hashes`` intersects with the last layer.

        If there is intersection, returns a state from intersection. Otherwise, returns ``None``.
        """
        mask = isin_via_searchsorted(self.hashes[-1], hashes)
        if not torch.any(mask):
            return None
        return self.graph.decode_states(self.cur_layer[mask.nonzero().reshape((-1,))])[0]
