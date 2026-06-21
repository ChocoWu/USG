"""Object Associator.

"""

import torch
import torch.nn as nn

from .ops import pairwise_cosine


class CNNFilter(nn.Module):
    """3-layer CNN, kernel 3×3.

    Treats the association matrix as a single-channel 2D image ``(B, 1, N, M)``,
    leveraging local detail to filter out redundant noise, and outputs same-size
    refined association **logits** (no sigmoid; supervised with BCEWithLogits).
    """

    def __init__(self, hidden: int = 64, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2  # same padding, keeps N×M unchanged
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size, padding=pad),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """a: ``(B, N, M)`` -> ``(B, N, M)``."""
        x = self.net(a.unsqueeze(1))  # (B,1,N,M)
        return x.squeeze(1)


class ObjectAssociator(nn.Module):
    """Build the object association matrix for one modality pair (src, tgt).

    In the full model each unordered modality pair {*, ⋄} holds its own instance (the
    projections are modality-pair specific). This module handles a single pair, which
    keeps it easy to unit-test and compose.

    Args:
        dim: object query dimension.
        filter_hidden: hidden channels of the CNN filter.
        use_linear: enable the modality-specific linear projections.
            When off, cosine is computed on the raw queries.
        use_filter: enable the CNN filter. When off, returns the
            bidirectional mean cosine directly.
    """

    def __init__(
        self,
        dim: int = 256,
        filter_hidden: int = 64,
        use_linear: bool = True,
        use_filter: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_linear = use_linear
        self.use_filter = use_filter
        if use_linear:
            self.src2tgt = nn.Linear(dim, dim)  # F_{*→⋄}
            self.tgt2src = nn.Linear(dim, dim)  # F_{⋄→*}
        if use_filter:
            self.filter = CNNFilter(filter_hidden)

    def forward(self, q_src: torch.Tensor, q_tgt: torch.Tensor):
        """Compute association logits.

        Args:
            q_src: ``(B, Ns, d)`` object queries of the source modality *.
            q_tgt: ``(B, Nt, d)`` object queries of the target modality ⋄.

        Returns:
            assoc: ``(B, Ns, Nt)`` association **logits** (supervised with BCEWithLogits;
                at inference, sigmoid then top-k / Hungarian matching).
            cos_bi: ``(B, Ns, Nt)`` pre-filter bidirectional mean cosine similarity
                (∈[-1,1], for analysis/ablation).
        """
        if q_src.dim() != 3 or q_tgt.dim() != 3:
            raise ValueError("q_src/q_tgt must be (B, N, d)")
        if q_src.size(0) != q_tgt.size(0) or q_src.size(-1) != q_tgt.size(-1):
            raise ValueError("batch and feature dim must match")

        if self.use_linear:
            a_s2t = pairwise_cosine(self.src2tgt(q_src), q_tgt)          # (B, Ns, Nt)
            a_t2s = pairwise_cosine(self.tgt2src(q_tgt), q_src)          # (B, Nt, Ns)
        else:
            a_s2t = pairwise_cosine(q_src, q_tgt)                        # (B, Ns, Nt)
            a_t2s = pairwise_cosine(q_tgt, q_src)                        # (B, Nt, Ns)

        cos_bi = 0.5 * (a_s2t + a_t2s.transpose(-1, -2))                 # (B, Ns, Nt)

        assoc = self.filter(cos_bi) if self.use_filter else cos_bi
        return assoc, cos_bi

    @torch.no_grad()
    def predict(self, q_src: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
        """Return association probabilities ``sigmoid(logits)`` ∈ (0,1), ``(B, Ns, Nt)``."""
        assoc, _ = self.forward(q_src, q_tgt)
        return torch.sigmoid(assoc)