"""Uniform encoder output contract consumed by the shared mask decoder & model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class EncodedModality:
    """Per-modality encoder output (uniform across text / image / video / point).

    feats_per_scale: list of (B, Li, d) flattened features, one per mask-decoder scale.
                     Visual: 3 spatial scales (coarse->fine). Text/point: H repeated
                     across scales (paper: "H_S across different scales").
    context_tokens:  (B, Lc, d) compact contextualized feature H̄ for the relation
                     decoder context H (eq. 19).
    feat_sizes:      [(Hi, Wi)...] spatial sizes for masked attention, or None
                     (text / point placeholder -> non-masked cross-attention).
    mask_features:   (B, d, H, W) high-res per-pixel embedding H_3 for mask
                     prediction, or None.
    feat_key_padding_masks: optional list of (B, Li) bool masks (True = pad), per scale.
    context_key_padding_mask: optional (B, Lc) bool mask for the context tokens.
    is_point: True for the 3D point modality -> mask decoder uses the point-mask path
              (per-scale point features serve as per-point embeddings; masks are (B,N,P)).
    num_frames: for video, the number of frames T folded into the batch (batch = B*T).
                Lets the model apply the temporal encoder F_temp across frames; None
                for non-video modalities.
    """

    feats_per_scale: List[torch.Tensor]
    context_tokens: torch.Tensor
    feat_sizes: Optional[List[Tuple[int, int]]] = None
    mask_features: Optional[torch.Tensor] = None
    feat_key_padding_masks: Optional[List[torch.Tensor]] = None
    context_key_padding_mask: Optional[torch.Tensor] = None
    is_point: bool = False
    num_frames: Optional[int] = None


def repeat_encoded(em: "EncodedModality", n: int) -> "EncodedModality":
    """Repeat each batch element n times (repeat_interleave dim 0) without re-encoding.

    Used to align a single image with each of n video frames in I-V multimodal
    training: the heavy encoder runs once (batch B); features are then repeated to B*n.
    """
    def rep(t):
        return t.repeat_interleave(n, dim=0) if t is not None else None

    return EncodedModality(
        feats_per_scale=[rep(f) for f in em.feats_per_scale],
        context_tokens=rep(em.context_tokens),
        feat_sizes=em.feat_sizes,
        mask_features=rep(em.mask_features),
        feat_key_padding_masks=[rep(m) for m in em.feat_key_padding_masks]
        if em.feat_key_padding_masks is not None else None,
        context_key_padding_mask=rep(em.context_key_padding_mask),
        is_point=em.is_point,
        num_frames=em.num_frames,
    )