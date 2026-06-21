"""Modality-specific Object Detection Head.

For each object query the head predicts (a) a category label (open-vocabulary,
with a "no object" label) and (b) a segmentation mask. Before prediction, queries
are enriched with complementary information from other modalities through the
cross-modal association matrices.

Equations:
  q^I_i = q^I_i + Σ_j A^{I<->*}_{i,j} q^*_j ,  * ∈ {V, D, S}      (cross-modal fusion)
  c̄^{o,I}_i = <q^I_i, text_emb(category names)>                  (open-vocab cls)
  m̄^I_i = sigmoid( MLP(Q^I) · H^{I⊤}_3 )                          (mask prediction)

"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP


class ObjectDetectionHead(nn.Module):
    """Modality-specific detection head (design shared across modalities)."""

    def __init__(
        self,
        dim: int = 256,
        mask_embed_layers: int = 3,
        logit_scale_init: float = math.log(1 / 0.07),  # CLIP-style temperature
    ):
        super().__init__()
        self.dim = dim
        # MLP(Q) producing the mask embedding (dotted with per-pixel features)
        self.mask_embed = MLP(dim, dim, dim, num_layers=mask_embed_layers)
        # project the query into the text-embedding space before cosine classification
        self.query_proj = nn.Linear(dim, dim)
        # learnable "no object" class embedding (appended as the last category)
        self.no_object_embed = nn.Parameter(torch.randn(dim))
        # learnable temperature for the cosine classifier
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    # ------------------------------------------------------------------ #
    # cross-modal fusion
    # ------------------------------------------------------------------ #
    def fuse_cross_modal(
        self,
        q_self: torch.Tensor,
        assoc: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """q^I_i += Σ_j A^{I<->*}_{i,j} q^*_j.

        Args:
            q_self: (B, N, d) queries of the current modality.
            assoc: optional dict ``{other_modality: (A, q_other)}`` where
                A is (B, N, N_other) association *weights* in [0,1] and
                q_other is (B, N_other, d). None/empty -> no fusion (single modality).
        """
        if not assoc:
            return q_self
        fused = q_self
        for a_weight, q_other in assoc.values():
            fused = fused + torch.bmm(a_weight, q_other)  # (B, N, d)
        return fused

    # ------------------------------------------------------------------ #
    # category classification (open-vocabulary)
    # ------------------------------------------------------------------ #
    def classify(self, q: torch.Tensor, class_text_embeddings: torch.Tensor) -> torch.Tensor:
        """Open-vocab category logits via cosine similarity to class-name embeddings.

        Args:
            q: (B, N, d) (fused) queries.
            class_text_embeddings: (C, d) category-name embeddings in the common space.

        Returns:
            logits: (B, N, C+1); the last column is the learnable "no object" class.
        """
        q = F.normalize(self.query_proj(q), dim=-1)
        emb = torch.cat([class_text_embeddings, self.no_object_embed[None]], dim=0)  # (C+1, d)
        emb = F.normalize(emb, dim=-1)
        return self.logit_scale.exp() * (q @ emb.t())  # (B, N, C+1)

    # ------------------------------------------------------------------ #
    # mask prediction
    # ------------------------------------------------------------------ #
    def predict_masks(self, q: torch.Tensor, pixel_embed: torch.Tensor) -> torch.Tensor:
        """m̄_i = sigmoid( MLP(Q) · H_3^T ).

        Args:
            q: (B, N, d) (fused) queries.
            pixel_embed: (B, d, H, W) highest-resolution per-pixel embeddings H_3.

        Returns:
            masks: (B, N, H, W) in [0,1].
        """
        mask_embed = self.mask_embed(q)  # (B, N, d)
        logits = torch.einsum("bnd,bdhw->bnhw", mask_embed, pixel_embed)
        return logits.sigmoid()

    def predict_point_masks(self, q: torch.Tensor, point_embed: torch.Tensor) -> torch.Tensor:
        """3D instance masks m̄_i = sigmoid(<MLP(Q), per-point embedding>).

        Args:
            q: (B, N, d) (fused) queries.
            point_embed: (B, P, d) per-point embeddings (finest point-decoder scale).

        Returns:
            masks: (B, N, P) in [0,1] over the P points.
        """
        mask_embed = self.mask_embed(q)  # (B, N, d)
        logits = torch.einsum("bnd,bpd->bnp", mask_embed, point_embed)
        return logits.sigmoid()

    # ------------------------------------------------------------------ #
    # full forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        q_self: torch.Tensor,
        class_text_embeddings: torch.Tensor,
        pixel_embed: Optional[torch.Tensor] = None,
        assoc: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        point_mask: bool = False,
    ):
        """Returns (cls_logits (B,N,C+1), masks or None, fused_q (B,N,d)).

        masks: (B,N,H,W) for the 2D path, (B,N,P) for the 3D point path (``point_mask=True``,
        ``pixel_embed`` is then (B,P,d)), or None when ``pixel_embed`` is None (text).
        """
        fused = self.fuse_cross_modal(q_self, assoc)
        cls_logits = self.classify(fused, class_text_embeddings)
        if pixel_embed is None:
            masks = None
        elif point_mask:
            masks = self.predict_point_masks(fused, pixel_embed)
        else:
            masks = self.predict_masks(fused, pixel_embed)
        return cls_logits, masks, fused

    @torch.no_grad()
    def predict_labels(self, q: torch.Tensor, class_text_embeddings: torch.Tensor) -> torch.Tensor:
        """Open-vocab inference: argmax category index in [0, C] (C == "no object")."""
        return self.classify(q, class_text_embeddings).argmax(dim=-1)