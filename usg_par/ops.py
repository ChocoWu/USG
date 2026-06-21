"""Shared tensor ops used across USG-Par modules."""

import torch


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between every row of ``x`` and every row of ``y``.

    Pairwise cosine similarity between two sets of queries.

    Args:
        x: ``(..., N, d)`` one set of vectors.
        y: ``(..., M, d)`` another set of vectors (leading batch dims must broadcast with x).
        eps: numerical-stability term, avoids dividing by a zero-norm vector.

    Returns:
        ``(..., N, M)`` similarity matrix, ``out[..., i, j] = cos(x_i, y_j)`` ∈ [-1, 1].
    """
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x @ y.transpose(-1, -2)