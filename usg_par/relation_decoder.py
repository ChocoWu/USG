"""Relation Decoder.

Given the top-k subject/object pairs from the RPC, build relationship queries and
decode the final predicate for each pair via a transformer that cross-attends to
the contextualized multimodal features H.

Key equations:
  Q_rel = [Q_sub + E_sub ; Q_obj + E_obj]                                   
  H     = [H_S ; H̄_I ; H̄_V ; H̄_D]   (only the present modalities)         
  X^rel_l = F^rel_CA(X^rel_{l-1}, H, H)
          = softmax(F_q(X^rel_{l-1})^T F_k(H)) F_v(H)                        

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .rpc import RPCOutput


def _ffn(dim: int, ffn_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
        nn.Dropout(dropout), nn.Linear(ffn_dim, dim),
    )


class RelationDecoderLayer(nn.Module):
    """One relation-decoder layer: cross-attn(H) -> self-attn -> FFN (Fig. 15, post-norm)."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.0, use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _ffn(dim, ffn_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x_rel: torch.Tensor, h: Optional[torch.Tensor],
                h_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # cross-attention to contextualized features H (eq. 10/20)
        if self.use_cross_attn:
            if h is None:
                raise ValueError("use_cross_attn=True requires context features H")
            ca, _ = self.cross(x_rel, h, h, key_padding_mask=h_key_padding_mask)
            x_rel = self.norm1(x_rel + ca)
        # self-attention among relationship queries
        sa, _ = self.self_attn(x_rel, x_rel, x_rel)
        x_rel = self.norm2(x_rel + sa)
        # FFN
        x_rel = self.norm3(x_rel + self.ffn(x_rel))
        return x_rel


class RelationDecoder(nn.Module):
    """Transformer relation decoder + predicate classifier (L_rel layers, dim 256)."""

    def __init__(
        self,
        dim: int = 256,
        num_predicates: int = 56,   # dataset-specific; PSG has 56 (overridden per config)
        num_layers: int = 6,        # L_rel = 6 (appendix E.2)
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        concat_pairs: bool = True,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.concat_pairs = concat_pairs
        if concat_pairs:
            self.rel_input_proj = nn.Linear(2 * dim, dim)  # 2d -> d (see ambiguity note)
        self.layers = nn.ModuleList(
            [RelationDecoderLayer(dim, num_heads, ffn_dim, dropout, use_cross_attn)
             for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(dim, num_predicates)

    def build_rel_queries(self, q_sub, e_sub, q_obj, e_obj) -> torch.Tensor:
        """Construct relationship queries Q_rel (eq. 9). Inputs each (B, k, d)."""
        sub = q_sub + e_sub      # Q_sub + E_sub (residual with the initial embedding)
        obj = q_obj + e_obj      # Q_obj + E_obj
        if self.concat_pairs:
            return self.rel_input_proj(torch.cat([sub, obj], dim=-1))  # (B, k, d)
        return sub + obj                                              # (B, k, d)

    def forward(self, q_rel: torch.Tensor, h: Optional[torch.Tensor] = None,
                h_key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: q_rel (B, k, d), h (B, L, d). Returns: (logits (B, k, P), X^rel_L (B, k, d))."""
        x = q_rel
        for layer in self.layers:
            x = layer(x, h, h_key_padding_mask)
        return self.classifier(x), x

    def decode(self, rpc_out: RPCOutput, h: Optional[torch.Tensor] = None,
               h_key_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: build Q_rel from an RPCOutput then decode predicates."""
        q_rel = self.build_rel_queries(rpc_out.q_sub, rpc_out.e_sub, rpc_out.q_obj, rpc_out.e_obj)
        return self.forward(q_rel, h, h_key_padding_mask)


def concat_context_features(feats):
    """Concatenate per-modality contextualized features along the token dim (eq. 19).

    Args:
        feats: list of (B, L_i, d) tensors for the present modalities.

    Returns:
        h: (B, sum_i L_i, d) concatenated context.
        key_padding_mask: None here (all tokens valid). Variable-length batching
            with padding is handled by the top-level model when needed.
    """
    if not feats:
        raise ValueError("at least one modality feature is required")
    return torch.cat(feats, dim=1), None