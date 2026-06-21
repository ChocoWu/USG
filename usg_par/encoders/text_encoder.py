"""Text Encoder: OpenCLIP text tower.

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .types import EncodedModality


class TextEncoder(nn.Module):
    """Wraps an OpenCLIP model's text tower.

    Args:
        clip_model: an OpenCLIP model (from ``build_openclip``).
        dim: common feature dimension (256).
        freeze: freeze the CLIP text transformer (only the projection is trained).
    """

    def __init__(self, clip_model: nn.Module, dim: int = 256, freeze: bool = True):
        super().__init__()
        self.clip = clip_model
        self.text_width = clip_model.token_embedding.weight.shape[1]   # 768
        self.embed_dim = clip_model.text_projection.shape[1]           # 768 (pooled)
        self.proj = nn.Linear(self.text_width, dim)        # token features -> d
        self.pooled_proj = nn.Linear(self.embed_dim, dim)  # pooled class emb -> d
        self.freeze = freeze
        if freeze:
            for p in self.clip.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    def _token_features(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Mirror OpenCLIP text forward but return per-token features (B, L, width)."""
        m = self.clip
        cast_dtype = m.transformer.get_cast_dtype()
        x = m.token_embedding(token_ids).to(cast_dtype)     # (B, L, width)
        x = x + m.positional_embedding.to(cast_dtype)
        x = m.transformer(x, attn_mask=m.attn_mask)         # batch_first=True
        x = m.ln_final(x)                                   # (B, L, width)
        return x

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: token_ids (B, L). Returns (H_S (B, L, d), key_padding_mask (B, L) bool)."""
        if self.freeze:
            with torch.no_grad():
                feats = self._token_features(token_ids)
        else:
            feats = self._token_features(token_ids)
        h_s = self.proj(feats.float())
        # CLIP pads with token id 0 after the EOT; mark those as padding (ignored in attn)
        key_padding_mask = token_ids == 0
        return h_s, key_padding_mask

    def encode(self, token_ids: torch.Tensor, num_scales: int = 3) -> EncodedModality:
        """Uniform encoder API. H_S is used across all scales (non-masked path)."""
        h_s, kpm = self.forward(token_ids)
        return EncodedModality(
            feats_per_scale=[h_s] * num_scales,
            context_tokens=h_s,
            feat_sizes=None,
            mask_features=None,
            feat_key_padding_masks=[kpm] * num_scales,
            context_key_padding_mask=kpm,
        )

    @torch.no_grad()
    def encode_class_names(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Pooled CLIP embeddings for class/predicate names -> (C, d).

        Used by the detection head / relation classifier for open-vocab cosine matching.
        """
        pooled = self.clip.encode_text(token_ids).float()   # (C, embed_dim)
        return self.pooled_proj(pooled)