"""Relation Proposal Constructor.

The RPC turns refined object queries into a small set of the most promising subject-object pairs, avoiding the infeasible O(N^2) exhaustive pairing.

"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .layers import MLP
from .ops import pairwise_cosine


@dataclass
class RPCOutput:
    """Outputs of the Relation Proposal Constructor.

    Shapes (B = batch, N = #object queries, k = #selected pairs, d = dim):
        pair_confidence: (B, N, N)  cosine pair-confidence matrix C (∈[-1,1]),
                                    rows = subjects, cols = objects; supervises L_pair.
        q_sub, q_obj:    (B, k, d)  refined subject/object queries of selected pairs.
        e_sub, e_obj:    (B, k, d)  initial projector embeddings of selected pairs
                                    (residual term E_sub/E_obj).
        sub_idx, obj_idx:(B, k)     object-query indices of the selected pairs.
        scores:          (B, k)     confidence of the selected pairs.
    """

    pair_confidence: torch.Tensor
    q_sub: torch.Tensor
    q_obj: torch.Tensor
    e_sub: torch.Tensor
    e_obj: torch.Tensor
    sub_idx: torch.Tensor
    obj_idx: torch.Tensor
    scores: torch.Tensor


class TwoWayRACLayer(nn.Module):
    """One two-way relation-aware cross-attention layer (Fig. 14).

    Two parallel streams (subject, object). Within a layer both cross-attentions
    read the *previous-layer* states (per eq. 8/17), then each stream runs
    self-attention and an FFN. Post-norm residual blocks, matching the
    "Add & Norm" boxes in Fig. 14.
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.0):
        super().__init__()

        def mha():
            return nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        def ffn():
            return nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(ffn_dim, dim),
            )

        # subject stream
        self.sub_cross, self.sub_self, self.sub_ffn = mha(), mha(), ffn()
        self.sub_norm1, self.sub_norm2, self.sub_norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        # object stream
        self.obj_cross, self.obj_self, self.obj_ffn = mha(), mha(), ffn()
        self.obj_norm1, self.obj_norm2, self.obj_norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x_sub: torch.Tensor, x_obj: torch.Tensor):
        # cross-attention: each stream queries the other; both read previous states
        s_ca, _ = self.sub_cross(x_sub, x_obj, x_obj)
        o_ca, _ = self.obj_cross(x_obj, x_sub, x_sub)
        x_sub = self.sub_norm1(x_sub + s_ca)
        x_obj = self.obj_norm1(x_obj + o_ca)
        # self-attention
        s_sa, _ = self.sub_self(x_sub, x_sub, x_sub)
        o_sa, _ = self.obj_self(x_obj, x_obj, x_obj)
        x_sub = self.sub_norm2(x_sub + s_sa)
        x_obj = self.obj_norm2(x_obj + o_sa)
        # FFN
        x_sub = self.sub_norm3(x_sub + self.sub_ffn(x_sub))
        x_obj = self.obj_norm3(x_obj + self.obj_ffn(x_obj))
        return x_sub, x_obj


def select_topk_pairs(c: torch.Tensor, k: int):
    """Top-k selection over the flattened pair-confidence matrix.

    Args:
        c: (B, N, M) pair confidence (subjects x objects).
        k: number of pairs to keep (clamped to N*M).

    Returns:
        scores: (B, k), sub_idx: (B, k), obj_idx: (B, k).
    """
    b, n, m = c.shape
    k = min(k, n * m)
    scores, flat_idx = c.reshape(b, n * m).topk(k, dim=-1)
    sub_idx = flat_idx // m
    obj_idx = flat_idx % m
    return scores, sub_idx, obj_idx


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather tokens of ``x`` (B, N, d) at indices ``idx`` (B, k) -> (B, k, d)."""
    d = x.size(-1)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d))


class RelationProposalConstructor(nn.Module):
    """Subject/object projectors + two-way RAC + top-k pair selection."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,        # L_RPC = 4 (appendix E.2)
        num_heads: int = 8,
        ffn_dim: int = 2048,
        top_k: int = 100,           # project default (see CLAUDE.md decisions)
        use_proj: bool = True,
        use_rac: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.use_proj = use_proj
        self.use_rac = use_rac
        if use_proj:
            self.sub_proj = MLP(dim, dim, dim, num_layers=2)
            self.obj_proj = MLP(dim, dim, dim, num_layers=2)
        if use_rac:
            self.layers = nn.ModuleList(
                [TwoWayRACLayer(dim, num_heads, ffn_dim) for _ in range(num_layers)]
            )

    def forward(self, obj_queries: torch.Tensor, top_k: Optional[int] = None) -> RPCOutput:
        """Args: obj_queries (B, N, d). Returns: RPCOutput."""
        if obj_queries.dim() != 3:
            raise ValueError("obj_queries should be (B, N, d)")
        k = top_k if top_k is not None else self.top_k

        # 1) subject / object projectors -> E_sub, E_obj (eq. X^sub_0 = E_sub, X^obj_0 = E_obj)
        if self.use_proj:
            e_sub = self.sub_proj(obj_queries)
            e_obj = self.obj_proj(obj_queries)
        else:  # ablation: use object queries directly as both embeddings
            e_sub = e_obj = obj_queries

        # 2) two-way relation-aware cross-attention
        x_sub, x_obj = e_sub, e_obj
        if self.use_rac:
            for layer in self.layers:
                x_sub, x_obj = layer(x_sub, x_obj)

        # 3) Pair Confidence Matrix  C = cos(X^sub_L, X^obj_L)
        c = pairwise_cosine(x_sub, x_obj)  # (B, N, N)

        # 4) top-k pair selection + gather refined queries and initial embeddings
        scores, sub_idx, obj_idx = select_topk_pairs(c, k)
        return RPCOutput(
            pair_confidence=c,
            q_sub=_gather_tokens(x_sub, sub_idx),
            q_obj=_gather_tokens(x_obj, obj_idx),
            e_sub=_gather_tokens(e_sub, sub_idx),
            e_obj=_gather_tokens(e_obj, obj_idx),
            sub_idx=sub_idx,
            obj_idx=obj_idx,
            scores=scores,
        )