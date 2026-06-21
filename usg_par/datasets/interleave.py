"""Interleaved multi-dataset iterator for joint training.

Yields ``(key, batch)`` for a fixed number of steps, picking which dataset to draw
from by **temperature sampling** (weight ∝ size**(1/T), T≈2 to avoid the largest
dataset dominating — "domain balancing"). Each underlying DataLoader is cycled when
exhausted.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


def temperature_weights(sizes: Dict[str, int], temperature: float = 2.0) -> Dict[str, float]:
    """weight_i ∝ size_i**(1/T), normalized to sum 1."""
    w = {k: float(n) ** (1.0 / temperature) for k, n in sizes.items()}
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


class InterleavedLoader:
    """Interleave several DataLoaders, sampling one per step by given weights."""

    def __init__(self, loaders: Dict[str, torch.utils.data.DataLoader],
                 weights: Dict[str, float], total_steps: int, seed: int = 0):
        self.loaders = loaders
        self.keys = list(loaders.keys())
        probs = np.array([weights[k] for k in self.keys], dtype=float)
        self.probs = probs / probs.sum()
        self.total_steps = total_steps
        self.rng = np.random.default_rng(seed)
        self._iters: Dict[str, object] = {}

    def _next(self, key: str):
        if key not in self._iters:
            self._iters[key] = iter(self.loaders[key])
        try:
            return next(self._iters[key])
        except StopIteration:
            self._iters[key] = iter(self.loaders[key])
            return next(self._iters[key])

    def __iter__(self):
        for _ in range(self.total_steps):
            key = self.keys[int(self.rng.choice(len(self.keys), p=self.probs))]
            yield key, self._next(key)

    def __len__(self) -> int:
        return self.total_steps