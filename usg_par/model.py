"""USG-Par assembly.

"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .associator import ObjectAssociator
from .detection_head import ObjectDetectionHead
from .encoders.types import EncodedModality
from .mask_decoder import SharedMaskDecoder, TemporalEncoder
from .relation_decoder import RelationDecoder, concat_context_features
from .rpc import RelationProposalConstructor, RPCOutput


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


@dataclass
class ModalityOutput:
    refined_query: torch.Tensor                 # (B, N, d) mask-decoder output
    mask_logits: Optional[torch.Tensor]         # (B, N, H, W) or None
    fused_query: torch.Tensor                   # (B, N, d) after cross-modal fusion
    cls_logits: torch.Tensor                    # (B, N, C+1)
    pred_masks: Optional[torch.Tensor]          # (B, N, H, W) or None
    rpc_out: RPCOutput
    relation_logits: torch.Tensor               # (B, k, P)


@dataclass
class USGOutput:
    per_modality: Dict[str, ModalityOutput] = field(default_factory=dict)
    associations: Dict[str, torch.Tensor] = field(default_factory=dict)  # pair_key -> logits


class USGParCore(nn.Module):
    """Encoder-agnostic core of USG-Par."""

    def __init__(
        self,
        modalities: Tuple[str, ...] = ("text", "image", "video", "point"),
        dim: int = 256,
        num_queries: int = 100,
        num_predicates: int = 56,
        num_scales: int = 3,
        mask_decoder_layers: int = 9,
        rpc_layers: int = 4,
        relation_layers: int = 6,
        top_k: int = 100,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.dim = dim
        self.num_scales = num_scales
        self.top_k = top_k

        # per-modality learnable object queries (weights NOT shared across modalities)
        self.query_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(num_queries, dim)) for m in modalities}
        )
        # shared mask decoder (weights shared across modalities)
        self.mask_decoder = SharedMaskDecoder(dim, mask_decoder_layers, num_scales)
        # temporal encoder F_temp (video): links object queries across frames (§3.2)
        self.temporal_encoder = TemporalEncoder(dim)
        # modality-specific detection heads
        self.det_heads = nn.ModuleDict({m: ObjectDetectionHead(dim) for m in modalities})
        # object associators per unordered modality pair (multimodal only)
        pairs = modality_pairs if modality_pairs is not None else list(combinations(modalities, 2))
        self.associators = nn.ModuleDict(
            {_pair_key(a, b): ObjectAssociator(dim) for a, b in pairs}
        )
        # shared RPC + relation decoder
        self.rpc = RelationProposalConstructor(dim, rpc_layers, top_k=top_k)
        self.relation_decoder = RelationDecoder(dim, num_predicates, relation_layers)

    # ------------------------------------------------------------------ #
    def _run_mask_decoder(self, mod: str, feat: EncodedModality, b: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q0 = self.query_embed[mod].unsqueeze(0).expand(b, -1, -1)
        refined, mask_logits, _ = self.mask_decoder(
            q0, feat.feats_per_scale, feat.feat_sizes, feat.mask_features,
            point_mask=feat.is_point,
            feat_key_padding_masks=feat.feat_key_padding_masks,
        )
        return refined, mask_logits

    def _build_context(self, feats: Dict[str, EncodedModality]):
        """Concatenate per-modality context tokens for the relation decoder (eq. 19)."""
        ctx_list, mask_list = [], []
        any_mask = False
        for m in feats:
            ctx = feats[m].context_tokens
            ctx_list.append(ctx)
            kpm = feats[m].context_key_padding_mask
            if kpm is None:
                kpm = torch.zeros(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
            else:
                any_mask = True
            mask_list.append(kpm)
        h, _ = concat_context_features(ctx_list)
        key_padding_mask = torch.cat(mask_list, dim=1) if any_mask else None
        return h, key_padding_mask

    # ------------------------------------------------------------------ #
    def forward(
        self,
        feats: Dict[str, EncodedModality],
        class_text_embeddings: Dict[str, torch.Tensor],
        top_k: Optional[int] = None,
    ) -> USGOutput:
        assert feats, "at least one modality required"
        out = USGOutput()

        # 1) shared mask decoder -> refined per-modality queries
        refined: Dict[str, torch.Tensor] = {}
        mask_logits: Dict[str, Optional[torch.Tensor]] = {}
        for m, feat in feats.items():
            mb = feat.feats_per_scale[0].size(0)            # encoder batch (B*T for video)
            refined[m], mask_logits[m] = self._run_mask_decoder(m, feat, mb)
            # temporal encoder F_temp: link object queries across video frames (§3.2)
            t = feat.num_frames
            if t and t > 1:
                bt, n, d = refined[m].shape
                r = self.temporal_encoder(refined[m].reshape(bt // t, t, n, d))
                refined[m] = r.reshape(bt, n, d)

        # 2) object associations between every present modality pair
        present = list(feats.keys())
        assoc_logits: Dict[str, torch.Tensor] = {}
        for a, b_mod in combinations(present, 2):
            key = _pair_key(a, b_mod)
            if key not in self.associators:
                continue
            # call with a deterministic src/tgt order matching the pair key
            s, t = sorted((a, b_mod))
            logits, _ = self.associators[key](refined[s], refined[t])
            assoc_logits[key] = logits
            out.associations[key] = logits

        # 3) shared relation context H (eq. 19)
        context_h, context_kpm = self._build_context(feats)

        # 4) per-modality detection + relation
        for m, feat in feats.items():
            # gather cross-modal association weights for fusion (sigmoid -> [0,1])
            assoc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for other in present:
                if other == m:
                    continue
                key = _pair_key(m, other)
                if key not in assoc_logits:
                    continue
                s, t = sorted((m, other))
                A = assoc_logits[key].sigmoid()
                # orient so rows correspond to modality m
                A = A if s == m else A.transpose(1, 2)
                assoc[other] = (A, refined[other])

            # mask source: 2D pixel embedding (visual), finest point features (3D), or none (text)
            if feat.is_point:
                mask_embed_src, point_mask = feat.feats_per_scale[-1], True
            else:
                mask_embed_src, point_mask = feat.mask_features, False
            cls_logits, pred_masks, fused = self.det_heads[m](
                refined[m], class_text_embeddings[m], mask_embed_src, assoc or None,
                point_mask=point_mask,
            )
            rpc_out = self.rpc(fused, top_k=top_k or self.top_k)
            relation_logits, _ = self.relation_decoder.decode(rpc_out, context_h, context_kpm)

            out.per_modality[m] = ModalityOutput(
                refined_query=refined[m], mask_logits=mask_logits[m], fused_query=fused,
                cls_logits=cls_logits, pred_masks=pred_masks, rpc_out=rpc_out,
                relation_logits=relation_logits,
            )
        return out


class USGPar(nn.Module):
    """Full model = real encoders + USGParCore."""

    def __init__(self, clip_model: nn.Module, core: Optional[USGParCore] = None,
                 point_pretrained: Optional[str] = None, point_in_dim: int = 6,
                 point_freeze_encoder: bool = False,
                 point_decoder_scales=(1024, 2048, None), **core_kwargs):
        super().__init__()
        from .encoders import ImageEncoder, PointEncoder, TextEncoder
        dim = core.dim if core is not None else core_kwargs.get("dim", 256)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = ImageEncoder(clip_model)
        self.point_encoder = PointEncoder(
            dim=dim, in_dim=point_in_dim, pretrained=point_pretrained,
            freeze_encoder=point_freeze_encoder, decoder_scales=point_decoder_scales)
        self.core = core or USGParCore(**core_kwargs)

    def encode_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, EncodedModality]:
        feats: Dict[str, EncodedModality] = {}
        if "text" in inputs:
            feats["text"] = self.text_encoder.encode(inputs["text"], self.core.num_scales)
        if "image" in inputs:
            feats["image"] = self.image_encoder.encode(inputs["image"])
        if "video" in inputs:  # (B,T,3,H,W); frames folded to batch B*T, temporal model TBD
            feats["video"] = self.image_encoder.encode_video(inputs["video"])
        if "point" in inputs:
            feats["point"] = self.point_encoder.encode(inputs["point"])
        return feats

    def forward(self, inputs: Dict[str, torch.Tensor],
                class_text_embeddings: Dict[str, torch.Tensor], top_k: Optional[int] = None) -> USGOutput:
        return self.core(self.encode_inputs(inputs), class_text_embeddings, top_k=top_k)

    def encode_iv(self, image: torch.Tensor, frames: torch.Tensor) -> Dict[str, EncodedModality]:
        """I-V encoding: encode the image ONCE (B) then repeat its features to B*T to
        align with each video frame — avoids re-running the backbone T times."""
        from .encoders.types import repeat_encoded
        t = frames.shape[1]
        img_em = repeat_encoded(self.image_encoder.encode(image), t)   # B -> B*T (features repeated)
        vid_em = self.image_encoder.encode_video(frames)               # B*T
        return {"image": img_em, "video": vid_em}