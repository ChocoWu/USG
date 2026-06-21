"""Shared Mask Decoder.

A Mask2Former-style cascaded transformer decoder, shared (same weights) across all modalities. 
Learnable object queries are refined by masked cross-attention to multi-scale modality features; a mask is predicted at every layer and binarized to form the attention mask of the next layer.

Equations:
  X*_l = softmax(M*_{l-1} + Q*_{l-1} K*_{l-1}^T) V*_{l-1} + X*_{l-1} 
         Q=Fq(X_{l-1}), K=Fk(H*), V=Fv(H*)
  M*_{l-1}(x,y) = 0 if mask==1 else -inf                                

"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP


class MaskPredictor(nn.Module):
    """mask_embed = MLP(query); mask logits = <mask_embed, per-pixel embedding H_3>."""

    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.mask_embed = MLP(dim, dim, dim, num_layers=num_layers)

    def forward(self, query: torch.Tensor, pixel_embed: torch.Tensor) -> torch.Tensor:
        """query (B,N,d), pixel_embed (B,d,H,W) -> mask logits (B,N,H,W)."""
        me = self.mask_embed(query)
        return torch.einsum("bnd,bdhw->bnhw", me, pixel_embed)

    def forward_points(self, query: torch.Tensor, point_embed: torch.Tensor) -> torch.Tensor:
        """query (B,N,d), point_embed (B,P,d) -> point mask logits (B,N,P) (3D path)."""
        me = self.mask_embed(query)
        return torch.einsum("bnd,bpd->bnp", me, point_embed)


def build_attn_mask(mask_logits: torch.Tensor, size: Tuple[int, int], num_heads: int) -> torch.Tensor:
    """Binarize & resize the predicted mask into a masked-attention mask.

    Args:
        mask_logits: (B, N, H, W) mask logits from the previous layer.
        size: (Hi, Wi) target feature resolution of the current scale.
        num_heads: attention heads (mask is expanded per head).

    Returns:
        bool attn_mask (B*num_heads, N, Hi*Wi); True == "do not attend" (background).
        Detached (the attention mask carries no gradient, as in Mask2Former).
    """
    b, n, _, _ = mask_logits.shape
    resized = F.interpolate(mask_logits, size=size, mode="bilinear", align_corners=False)
    attn = (resized.sigmoid() < 0.5)              # True = background = disallow
    attn = attn.flatten(2)                        # (B, N, Hi*Wi)
    # safeguard: a query whose mask is entirely background would mask the whole row
    # (-> NaN softmax); let it attend everywhere instead.
    all_masked = attn.all(dim=-1, keepdim=True)
    attn = attn & ~all_masked
    # expand to (B*num_heads, N, L)
    attn = attn.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(b * num_heads, n, -1)
    return attn.detach()


def build_point_attn_mask(point_mask_logits: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Point-cloud masked-attention mask (3D analogue of build_attn_mask, no 2D grid).

    Args:
        point_mask_logits: (B, N, P) mask logits over the current scale's points.
        num_heads: attention heads.

    Returns:
        bool attn_mask (B*num_heads, N, P); True == "do not attend" (background), detached.
    """
    attn = (point_mask_logits.sigmoid() < 0.5)        # True = background
    all_masked = attn.all(dim=-1, keepdim=True)
    attn = attn & ~all_masked                          # empty-query safeguard
    b, n, p = attn.shape
    attn = attn.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(b * num_heads, n, p)
    return attn.detach()


class MaskDecoderLayer(nn.Module):
    """One Mask2Former decoder layer: masked cross-attn -> self-attn -> FFN (post-norm)."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.cross = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(ffn_dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query, feat, attn_mask=None, feat_key_padding_mask=None):
        ca, _ = self.cross(query, feat, feat, attn_mask=attn_mask,
                           key_padding_mask=feat_key_padding_mask)
        query = self.norm1(query + ca)
        sa, _ = self.self_attn(query, query, query)
        query = self.norm2(query + sa)
        query = self.norm3(query + self.ffn(query))
        return query


class SharedMaskDecoder(nn.Module):
    """Cascaded multi-scale mask decoder (shared across modalities)."""

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 9,        # L_mask
        num_scales: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [MaskDecoderLayer(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.mask_predictor = MaskPredictor(dim)
        self.decoder_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_init: torch.Tensor,                 # (B, N, d)  per-modality learnable queries
        feats_per_scale: List[torch.Tensor],      # list of (B, Li, d) flattened features
        feat_sizes: Optional[List[Tuple[int, int]]] = None,  # [(Hi,Wi)...] for visual; None for text/point
        mask_features: Optional[torch.Tensor] = None,        # (B, d, H, W) for visual mask prediction
        point_mask: bool = False,                            # enable 3D point-mask path
        feat_key_padding_masks: Optional[List[torch.Tensor]] = None,
    ):
        """Returns (refined_query (B,N,d), mask_logits or None, intermediate_masks list).

        Mask modes: spatial (visual, 2D masked attn) > point (3D, point masked attn) > none (text).
        For the point path the per-scale point features double as the per-point embeddings;
        the final mask is predicted at the finest scale (full point resolution).
        """
        spatial = mask_features is not None and feat_sizes is not None
        point = (not spatial) and point_mask
        x = query_init
        intermediate = []

        mask_logits = self.mask_predictor(x, mask_features) if spatial else None

        for l in range(self.num_layers):
            s = l % self.num_scales
            feat = feats_per_scale[s]
            kpm = feat_key_padding_masks[s] if feat_key_padding_masks is not None else None
            if spatial:
                attn_mask = build_attn_mask(mask_logits, feat_sizes[s], self.num_heads)
            elif point:
                m_s = self.mask_predictor.forward_points(x, feat)   # predict at scale s
                attn_mask = build_point_attn_mask(m_s, self.num_heads)
            else:
                attn_mask = None
            x = self.layers[l](x, feat, attn_mask=attn_mask, feat_key_padding_mask=kpm)
            if spatial:
                mask_logits = self.mask_predictor(x, mask_features)
                intermediate.append(mask_logits)
            elif point:
                mask_logits = self.mask_predictor.forward_points(x, feats_per_scale[-1])  # full-res
                intermediate.append(mask_logits)

        x = self.decoder_norm(x)
        return x, mask_logits, intermediate


class TemporalEncoder(nn.Module):
    """Transformer temporal encoder F_temp for video (2 layers).

    Models temporal relationships between objects across frames: for each object slot,
    attend across the T frames.
    """

    def __init__(self, dim: int = 256, num_layers: int = 2, num_heads: int = 8, ffn_dim: int = 2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=ffn_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, frame_queries: torch.Tensor) -> torch.Tensor:
        """frame_queries (B, T, N, d) -> temporally-encoded (B, T, N, d)."""
        b, t, n, d = frame_queries.shape
        # attend over the time dimension per object slot
        x = frame_queries.permute(0, 2, 1, 3).reshape(b * n, t, d)
        x = self.encoder(x)
        return x.reshape(b, n, t, d).permute(0, 2, 1, 3).contiguous()