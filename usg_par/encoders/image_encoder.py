"""Image / Video Encoder: frozen CLIP-ConvNeXt-L backbone + pixel decoder.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import EncodedModality


# --------------------------------------------------------------------------- #
# 2D sine positional embedding
# --------------------------------------------------------------------------- #
def sine_pos_embed_2d(h: int, w: int, dim: int, device, temperature: float = 10000.0) -> torch.Tensor:
    """Return (h*w, dim) 2D sine positional embedding."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sine pos embed"
    d = dim // 2
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(d // 2, device=device).float()
    omega = 1.0 / (temperature ** (omega / (d // 2)))
    out_x = xx.flatten()[:, None] * omega[None, :]
    out_y = yy.flatten()[:, None] * omega[None, :]
    pe = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pe  # (h*w, dim)


@dataclass
class VisualFeatures:
    """Outputs of the image/video encoder, matching SharedMaskDecoder's interface.

    feats_per_scale: list of (B, Li, d) flattened features (coarse -> fine).
    feat_sizes:      list of (Hi, Wi) per scale.
    mask_features:   (B, d, H, W) highest-resolution per-pixel embedding H_3.
    context_tokens:  (B, Lc, d) compact contextualized feature H̄ for the
                     relation-decoder context (eq. 19); the coarsest scale.
    """

    feats_per_scale: List[torch.Tensor]
    feat_sizes: List[Tuple[int, int]]
    mask_features: torch.Tensor
    context_tokens: torch.Tensor


class ConvNeXtBackbone(nn.Module):
    """Frozen CLIP-ConvNeXt-L trunk returning the last 3 stage feature maps."""

    def __init__(self, clip_model: nn.Module, freeze: bool = True):
        super().__init__()
        self.trunk = clip_model.visual.trunk  # timm ConvNeXt
        # channels of the 4 stages: [192, 384, 768, 1536]; we use the last 3.
        self.out_channels = [f["num_chs"] for f in self.trunk.feature_info][1:]
        self.freeze = freeze
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x (B,3,H,W) -> [s8 (B,384,..), s16 (B,768,..), s32 (B,1536,..)] (fine->coarse)."""
        def run():
            y = self.trunk.stem(x)
            feats = []
            for stage in self.trunk.stages:
                y = stage(y)
                feats.append(y)
            return feats[1:]  # drop stride-4 stage, keep s8/s16/s32

        if self.freeze:
            with torch.no_grad():
                return run()
        return run()


class PixelDecoder(nn.Module):
    """Deformable-free multi-scale transformer pixel decoder (see module docstring)."""

    def __init__(
        self,
        in_channels: List[int],          # [384, 768, 1536] (fine->coarse from backbone)
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        mask_feature_size_scale: int = 2,  # upsample finest map by this factor for H_3
    ):
        super().__init__()
        self.dim = dim
        self.mask_feature_size_scale = mask_feature_size_scale
        self.input_proj = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_channels])
        self.level_embed = nn.Parameter(torch.randn(len(in_channels), dim))
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.mask_proj = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> VisualFeatures:
        """feats: [s8, s16, s32] (fine->coarse). Returns VisualFeatures (coarse->fine)."""
        b = feats[0].size(0)
        device = feats[0].device
        projected = [proj(f) for proj, f in zip(self.input_proj, feats)]  # each (B,d,Hi,Wi)
        sizes = [(f.shape[-2], f.shape[-1]) for f in projected]

        # flatten + add level & positional embeddings, concat across scales
        tokens, splits = [], []
        for lvl, f in enumerate(projected):
            h, w = f.shape[-2:]
            t = f.flatten(2).transpose(1, 2)                     # (B, Hi*Wi, d)
            pos = sine_pos_embed_2d(h, w, self.dim, device)[None]  # (1, Hi*Wi, d)
            t = t + pos + self.level_embed[lvl][None, None]
            tokens.append(t)
            splits.append(h * w)
        fused = self.encoder(torch.cat(tokens, dim=1))           # (B, sum, d)

        # split back to per-scale maps
        per_scale_maps = []
        offset = 0
        for (h, w), n in zip(sizes, splits):
            chunk = fused[:, offset:offset + n].transpose(1, 2).reshape(b, self.dim, h, w)
            per_scale_maps.append(chunk)
            offset += n

        # high-res mask features H_3 from the finest map (index 0 = s8)
        finest = per_scale_maps[0]
        up = F.interpolate(finest, scale_factor=self.mask_feature_size_scale,
                           mode="bilinear", align_corners=False)
        mask_features = self.mask_proj(up)                       # (B, d, H, W)

        # order scales coarse -> fine for the mask decoder's round-robin (Mask2Former style)
        order = list(reversed(range(len(per_scale_maps))))       # [s32, s16, s8]
        feats_per_scale = [per_scale_maps[i].flatten(2).transpose(1, 2) for i in order]
        feat_sizes = [sizes[i] for i in order]
        context_tokens = feats_per_scale[0]                      # coarsest = compact H̄

        return VisualFeatures(feats_per_scale, feat_sizes, mask_features, context_tokens)


class ImageEncoder(nn.Module):
    """Frozen ConvNeXt-L backbone + trainable pixel decoder."""

    def __init__(self, clip_model: nn.Module, dim: int = 256, freeze_backbone: bool = True,
                 pixel_decoder_layers: int = 4):
        super().__init__()
        self.backbone = ConvNeXtBackbone(clip_model, freeze=freeze_backbone)
        self.pixel_decoder = PixelDecoder(
            self.backbone.out_channels, dim=dim, num_layers=pixel_decoder_layers
        )

    def forward(self, images: torch.Tensor) -> VisualFeatures:
        """images (B, 3, H, W) -> VisualFeatures."""
        return self.pixel_decoder(self.backbone(images))

    def encode(self, images: torch.Tensor) -> EncodedModality:
        """Uniform encoder API -> EncodedModality."""
        vf = self.forward(images)
        return EncodedModality(
            feats_per_scale=vf.feats_per_scale,
            context_tokens=vf.context_tokens,
            feat_sizes=vf.feat_sizes,
            mask_features=vf.mask_features,
        )

    def forward_video(self, video: torch.Tensor) -> Tuple[VisualFeatures, int, int]:
        """Encode a video by folding frames into the batch.

        Args:
            video: (B, T, 3, H, W).

        Returns:
            (VisualFeatures with batch B*T, B, T). Each frame is encoded independently
            (per-frame VSG); temporal aggregation of the resulting object queries is
            handled later by the model's TemporalEncoder.
        """
        if video.dim() != 5:
            raise ValueError("video must be (B, T, 3, H, W)")
        b, t = video.shape[:2]
        frames = video.reshape(b * t, *video.shape[2:])
        return self.forward(frames), b, t

    def encode_video(self, video: torch.Tensor) -> EncodedModality:
        """Uniform encoder API for video -> EncodedModality (batch folded to B*T).

        Sets num_frames=T so the model can apply the temporal encoder F_temp.
        """
        vf, _, t = self.forward_video(video)
        return EncodedModality(
            feats_per_scale=vf.feats_per_scale,
            context_tokens=vf.context_tokens,
            feat_sizes=vf.feat_sizes,
            mask_features=vf.mask_features,
            num_frames=t,
        )