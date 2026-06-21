"""Point Encoder + Point Decoder, Point-BERT PointTransformer style.

Encoder (Point-BERT, https://github.com/Julie-tang00/Point-BERT):
  Group (FPS centers + KNN neighborhoods) -> mini-PointNet Encoder -> super-point
  tokens; a transformer with a positional embedding from the center coordinates
  refines them. Config (PointTransformer_8192point.yaml): num_group=512,
  group_size=32, encoder_dims=256, trans_dim=384, depth=12, num_heads=6.

Decoder (paper: "hierarchical propagation strategy with distance-based
  interpolation, producing multi-scale point features {H_D}_3, i=3 = original
  points"): PointNet++ feature-propagation — 3-NN inverse-distance interpolation
  from super-points to progressively finer point sets, 3 scales (coarse->fine).

"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..layers import MLP
from .types import EncodedModality


# --------------------------------------------------------------------------- #
# point-set ops (pure PyTorch)
# --------------------------------------------------------------------------- #
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points (B,N,C), idx (B,...) -> (B,...,C)."""
    b = points.size(0)
    view = [b] + [1] * (idx.dim() - 1)
    rep = [1] + list(idx.shape[1:])
    batch_idx = torch.arange(b, device=points.device).view(view).repeat(rep)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative FPS (deterministic, starts at index 0). xyz (B,N,3) -> idx (B,npoint)."""
    b, n, _ = xyz.shape
    device = xyz.device
    npoint = min(npoint, n)
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch_idx = torch.arange(b, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1)[1]
    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """k nearest neighbors. xyz (B,N,3) source, new_xyz (B,S,3) query -> idx (B,S,k)."""
    dist = torch.cdist(new_xyz, xyz)  # (B,S,N)
    return dist.topk(min(k, xyz.size(1)), dim=-1, largest=False)[1]


def three_nn_interpolate(target_xyz, source_xyz, source_feat) -> torch.Tensor:
    """Inverse-distance 3-NN interpolation (PointNet++ FP).

    target_xyz (B,Nt,3), source_xyz (B,Ns,3), source_feat (B,Ns,C) -> (B,Nt,C).
    """
    dist = torch.cdist(target_xyz, source_xyz)            # (B,Nt,Ns)
    d, idx = dist.topk(min(3, source_xyz.size(1)), dim=-1, largest=False)  # (B,Nt,3)
    weight = 1.0 / d.clamp_min(1e-10)
    weight = weight / weight.sum(-1, keepdim=True)        # (B,Nt,3)
    gathered = index_points(source_feat, idx)             # (B,Nt,3,C)
    return (gathered * weight.unsqueeze(-1)).sum(2)       # (B,Nt,C)


# --------------------------------------------------------------------------- #
# Point-BERT Group + Encoder
# --------------------------------------------------------------------------- #
class Group(nn.Module):
    """FPS centers + KNN neighborhoods (Point-BERT dvae.Group)."""

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor):
        """xyz (B,N,3) -> (neighborhood_xyz (B,G,M,3) normalized, center (B,G,3), idx (B,G,M))."""
        center_idx = farthest_point_sample(xyz, self.num_group)   # (B,G)
        center = index_points(xyz, center_idx)                    # (B,G,3)
        idx = knn_point(self.group_size, xyz, center)             # (B,G,M)
        neighborhood = index_points(xyz, idx)                     # (B,G,M,3)
        neighborhood = neighborhood - center.unsqueeze(2)         # normalize
        return neighborhood, center, idx


class PointNetEncoder(nn.Module):
    """Two-stage mini-PointNet group embedding (Point-BERT dvae.Encoder)."""

    def __init__(self, encoder_channel: int, in_dim: int = 3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """point_groups (B,G,M,in_dim) -> group tokens (B,G,encoder_channel)."""
        bs, g, m, c = point_groups.shape
        x = point_groups.reshape(bs * g, m, c).transpose(2, 1)    # (BG, in_dim, M)
        feat = self.first_conv(x)                                 # (BG, 256, M)
        feat_global = feat.max(dim=2, keepdim=True)[0]            # (BG, 256, 1)
        feat = torch.cat([feat_global.expand(-1, -1, m), feat], dim=1)  # (BG, 512, M)
        feat = self.second_conv(feat)                            # (BG, C, M)
        feat_global = feat.max(dim=2)[0]                         # (BG, C)
        return feat_global.reshape(bs, g, self.encoder_channel)


# --------------------------------------------------------------------------- #
# transformer (Point-BERT style: pos added before every block)
# --------------------------------------------------------------------------- #
# Module names below mirror Point-BERT (attn.qkv/proj, mlp.fc1/fc2, blocks.blocks.N)
# so the released PointTransformer checkpoint loads directly (see load_point_bert).
class _Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # Point-BERT: qkv has no bias
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim, num_heads) for _ in range(depth)])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)   # Point-BERT adds positional embedding before each block
        return x


# --------------------------------------------------------------------------- #
# Point Transformer encoder (super-point features)
# --------------------------------------------------------------------------- #
class PointTransformerEncoder(nn.Module):
    def __init__(self, in_dim=6, num_group=512, group_size=32, encoder_dims=256,
                 trans_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.group = Group(num_group, group_size)
        self.encoder = PointNetEncoder(encoder_dims, in_dim=in_dim)
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )
        self.blocks = _TransformerEncoder(trans_dim, depth, num_heads)   # -> blocks.blocks.N
        self.norm = nn.LayerNorm(trans_dim)
        self.in_dim = in_dim

    def load_point_bert(self, ckpt_path: str, verbose: bool = True):
        """Load released Point-BERT / PointTransformer weights into this encoder.

        Handles the ``module.`` DataParallel prefix and drops the ModelNet
        classification head (``cls_head_finetune``). Loads with strict=False so any
        shape mismatch (e.g. encoder.first_conv when in_dim != 3) is reported, not fatal.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("base_model", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        clean = {}
        for k, v in sd.items():
            k = k[len("module."):] if k.startswith("module.") else k
            if k.startswith("cls_head_finetune"):   # ModelNet classifier, not needed
                continue
            clean[k] = v
        missing, unexpected = self.load_state_dict(clean, strict=False)
        if verbose:
            loaded = len(clean) - len(unexpected)
            print(f"[Point-BERT] loaded {loaded}/{len(clean)} tensors | "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        return missing, unexpected

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """points (B,P,in_dim) [xyz(+rgb)] -> (super-point tokens (B,G,trans_dim), centers (B,G,3))."""
        xyz = points[..., :3]
        neigh_xyz, center, idx = self.group(xyz)                 # (B,G,M,3),(B,G,3),(B,G,M)
        if self.in_dim > 3:
            extra = index_points(points[..., 3:], idx)           # (B,G,M,in_dim-3)
            groups = torch.cat([neigh_xyz, extra], dim=-1)       # (B,G,M,in_dim)
        else:
            groups = neigh_xyz
        tokens = self.reduce_dim(self.encoder(groups))           # (B,G,trans_dim)

        b = tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        cls_pos = self.cls_pos.expand(b, -1, -1)
        pos = self.pos_embed(center)                             # (B,G,trans_dim)
        x = torch.cat([cls, tokens], dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)
        x = self.norm(self.blocks(x, pos))
        return x[:, 1:], center                                  # drop cls -> super-point tokens


# --------------------------------------------------------------------------- #
# Point decoder (hierarchical FP)
# --------------------------------------------------------------------------- #
class PointDecoder(nn.Module):
    """Hierarchical distance-based interpolation to 3 scales (coarse->fine)."""

    def __init__(self, in_dim: int, dim: int = 256,
                 scale_points: Tuple[Optional[int], ...] = (1024, 2048, None)):
        super().__init__()
        # scale_points: coarse -> fine; None == full original resolution.
        self.scale_points = scale_points
        self.proj_in = nn.Linear(in_dim, dim)
        self.fp_mlps = nn.ModuleList([MLP(dim + 3, dim, dim, num_layers=2) for _ in scale_points])

    def forward(self, center_xyz, center_feat, points_xyz) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns list of (xyz (B,ni,3), feat (B,ni,dim)) coarse->fine."""
        n = points_xyz.size(1)
        prev_xyz, prev_feat = center_xyz, self.proj_in(center_feat)
        outs = []
        for i, npts in enumerate(self.scale_points):
            if npts is None or npts >= n:
                tgt_xyz = points_xyz
            else:
                tgt_xyz = index_points(points_xyz, farthest_point_sample(points_xyz, npts))
            interp = three_nn_interpolate(tgt_xyz, prev_xyz, prev_feat)        # (B,ni,dim)
            feat = self.fp_mlps[i](torch.cat([interp, tgt_xyz], dim=-1))       # (B,ni,dim)
            outs.append((tgt_xyz, feat))
            prev_xyz, prev_feat = tgt_xyz, feat
        return outs


# --------------------------------------------------------------------------- #
# Combined Point Encoder (encoder + decoder -> EncodedModality)
# --------------------------------------------------------------------------- #
class PointEncoder(nn.Module):
    def __init__(self, dim: int = 256, in_dim: int = 6, num_group: int = 512,
                 group_size: int = 32, encoder_dims: int = 256, trans_dim: int = 384,
                 depth: int = 12, num_heads: int = 6,
                 decoder_scales: Tuple[Optional[int], ...] = (1024, 2048, None),
                 pretrained: Optional[str] = None, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = PointTransformerEncoder(
            in_dim, num_group, group_size, encoder_dims, trans_dim, depth, num_heads
        )
        self.decoder = PointDecoder(trans_dim, dim, decoder_scales)
        if pretrained is not None:   # load Point-BERT weights into the encoder
            self.encoder.load_point_bert(pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, points: torch.Tensor) -> EncodedModality:
        """points (B,P,in_dim) -> EncodedModality (3 point-feature scales, coarse->fine)."""
        tokens, centers = self.encoder(points)                  # (B,G,trans_dim),(B,G,3)
        scale_feats = self.decoder(centers, tokens, points[..., :3])
        feats_per_scale = [f for _, f in scale_feats]
        # finest per-point embedding (feats_per_scale[-1]) is the H_3 analog for a
        # future 3D point-mask head; the shared mask decoder currently uses the
        # non-masked path for 3D (mask_features=None).
        return EncodedModality(
            feats_per_scale=feats_per_scale,
            context_tokens=feats_per_scale[0],                  # coarsest = compact context
            feat_sizes=None,
            mask_features=None,
            is_point=True,                                      # -> point-mask path in mask decoder
        )

    encode = forward