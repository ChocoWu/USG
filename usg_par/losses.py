"""Training objectives.

Implements the four loss groups and their combination:

  L_obj  = λ_cls·L^o_cls + λ_ce·L_ce + λ_dice·L_dice          
           with Hungarian matching between predicted and GT entity masks.
  L_ass  = weighted BCE on the GT binary association matrix.
  L_rel  = L^r_cls + L_pair                                   
  L_cons = L^o_cons + L^r_cons   (text-centric contrastive)
  L      = α·L_obj + β·L_ass + γ·L_rel + η·L_cons            
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EPS = 1e-6


# =========================================================================== #
# Mask losses
# =========================================================================== #
def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss. pred (M, P) probabilities in [0,1], target (M, P) binary.

    Returns the mean over the M matched masks.
    """
    pred = pred.flatten(1)
    target = target.flatten(1)
    num = 2 * (pred * target).sum(-1)
    den = pred.sum(-1) + target.sum(-1)
    return (1 - (num + EPS) / (den + EPS)).mean()


def sigmoid_ce_mask_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary CE on mask *probabilities* (head already applied sigmoid). Mean over masks."""
    return F.binary_cross_entropy(pred.flatten(1).clamp(EPS, 1 - EPS),
                                  target.flatten(1), reduction="mean")


# =========================================================================== #
# Hungarian matcher (per image)
# =========================================================================== #
@dataclass
class MatcherWeights:
    w_class: float = 2.0
    w_mask: float = 5.0
    w_dice: float = 5.0


class HungarianMatcher:
    """Bipartite matching of predicted queries to GT entities (DETR/Mask2Former style)."""

    def __init__(self, weights: Optional[MatcherWeights] = None):
        self.w = weights or MatcherWeights()

    @torch.no_grad()
    def match_one(self, cls_logits, pred_masks, gt_labels, gt_masks) -> Tuple[torch.Tensor, torch.Tensor]:
        """One sample.

        Args:
            cls_logits: (N, C+1) category logits.
            pred_masks: (N, H, W) probabilities, or None (text modality).
            gt_labels:  (M,) GT class indices.
            gt_masks:   (M, H, W) binary, or None.

        Returns:
            (pred_idx, gt_idx) LongTensors of matched indices.
        """
        n = cls_logits.size(0)
        m = gt_labels.size(0)
        if m == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty

        prob = cls_logits.sigmoid()                       # (N, C+1)
        cost_class = -prob[:, gt_labels]                  # (N, M)
        cost = self.w.w_class * cost_class

        if pred_masks is not None and gt_masks is not None:
            p = pred_masks.flatten(1).clamp(EPS, 1 - EPS)  # (N, P)
            t = gt_masks.flatten(1).float()                # (M, P)
            num_px = p.size(1)
            # pairwise BCE
            pos = -torch.log(p) @ t.t()                    # (N, M)
            neg = -torch.log(1 - p) @ (1 - t).t()          # (N, M)
            cost_mask = (pos + neg) / num_px
            # pairwise dice
            num = 2 * (p @ t.t())
            den = p.sum(-1, keepdim=True) + t.sum(-1)[None, :]
            cost_dice = 1 - (num + EPS) / (den + EPS)
            cost = cost + self.w.w_mask * cost_mask + self.w.w_dice * cost_dice

        pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(gt_idx, dtype=torch.long)


# =========================================================================== #
# Object detection loss (eq. 11)
# =========================================================================== #
@dataclass
class DetectionLossWeights:
    matched_class: float = 2.0   # λ_cls for matched queries
    no_object: float = 0.1       # weight for the "no object" target
    ce: float = 5.0              # λ_ce
    dice: float = 5.0            # λ_dice


def object_detection_loss(
    cls_logits: torch.Tensor,        # (B, N, C+1)
    pred_masks: Optional[torch.Tensor],  # (B, N, H, W) or None
    gt_labels: List[torch.Tensor],   # list of (M_b,)
    gt_masks: Optional[List[torch.Tensor]],  # list of (M_b, H, W) or None
    matcher: HungarianMatcher,
    weights: Optional[DetectionLossWeights] = None,
    return_matches: bool = False,
):
    """Compute L_obj with Hungarian matching. Returns dict of scalar losses.

    If ``return_matches`` is True the dict also contains ``matches``: a per-sample
    list of (pred_idx, gt_idx) LongTensors (used to build relation targets).
    """
    w = weights or DetectionLossWeights()
    b, n, c1 = cls_logits.shape
    no_obj = c1 - 1  # index of the "no object" class

    cls_loss = cls_logits.new_zeros(())
    ce_loss = cls_logits.new_zeros(())
    dice_loss_val = cls_logits.new_zeros(())
    n_mask_terms = 0
    matches = []

    for bi in range(b):
        masks_b = pred_masks[bi] if pred_masks is not None else None
        gtm_b = gt_masks[bi] if gt_masks is not None else None
        pred_idx, gt_idx = matcher.match_one(cls_logits[bi], masks_b, gt_labels[bi], gtm_b)
        matches.append((pred_idx, gt_idx))

        # --- classification: per-query weighted sigmoid CE over C+1 ---
        target = cls_logits.new_full((n,), no_obj, dtype=torch.long)
        target[pred_idx] = gt_labels[bi][gt_idx].to(target.device)
        target_oh = F.one_hot(target, c1).float()
        per_query = F.binary_cross_entropy_with_logits(
            cls_logits[bi], target_oh, reduction="none").mean(-1)  # (N,)
        q_weight = cls_logits.new_full((n,), w.no_object)
        q_weight[pred_idx] = w.matched_class
        cls_loss = cls_loss + (per_query * q_weight).sum() / q_weight.sum()

        # --- mask losses (only on matched queries) ---
        if masks_b is not None and gtm_b is not None and pred_idx.numel() > 0:
            mp = masks_b[pred_idx]
            mt = gtm_b[gt_idx].float()
            ce_loss = ce_loss + sigmoid_ce_mask_loss(mp, mt)
            dice_loss_val = dice_loss_val + dice_loss(mp, mt)
            n_mask_terms += 1

    cls_loss = cls_loss / b
    if n_mask_terms > 0:
        ce_loss = ce_loss / n_mask_terms
        dice_loss_val = dice_loss_val / n_mask_terms

    total = cls_loss + w.ce * ce_loss + w.dice * dice_loss_val
    out = {"cls": cls_loss, "ce": ce_loss, "dice": dice_loss_val, "total": total}
    if return_matches:
        out["matches"] = matches
    return out


# =========================================================================== #
# Association loss & pair loss (weighted BCE on sparse binary matrices)
# =========================================================================== #
def _auto_pos_weight(target: torch.Tensor, fixed: Optional[float]) -> torch.Tensor:
    """pos_weight ≈ #neg / #pos (CLAUDE.md decision), or a fixed value."""
    if fixed is not None:
        return target.new_tensor(float(fixed))
    pos = target.sum().clamp_min(1.0)
    neg = target.numel() - target.sum()
    return (neg / pos).clamp_min(1.0)


def association_loss(assoc_logits: torch.Tensor, gt_assoc: torch.Tensor,
                     pos_weight: Optional[float] = None) -> torch.Tensor:
    """Weighted BCE on the GT binary association matrix (L_ass). Inputs are logits."""
    pw = _auto_pos_weight(gt_assoc, pos_weight)
    return F.binary_cross_entropy_with_logits(assoc_logits, gt_assoc.float(), pos_weight=pw)


def pair_loss(pair_confidence: torch.Tensor, gt_pairs: torch.Tensor,
              pos_weight: Optional[float] = None) -> torch.Tensor:
    """Weighted BCE on the pair confidence matrix C (L_pair). C treated as logits."""
    pw = _auto_pos_weight(gt_pairs, pos_weight)
    return F.binary_cross_entropy_with_logits(pair_confidence, gt_pairs.float(), pos_weight=pw)


# =========================================================================== #
# Relation classification loss (eq. 12)
# =========================================================================== #
def predicate_loss(pred_logits: torch.Tensor, target_multihot: torch.Tensor) -> torch.Tensor:
    """Sigmoid CE for predicate classification (L^r_cls). pred_logits (B,k,P)."""
    return F.binary_cross_entropy_with_logits(pred_logits, target_multihot.float())


def relation_loss(pred_logits, target_multihot, pair_confidence, gt_pairs,
                  pair_pos_weight: Optional[float] = None):
    """L_rel = L^r_cls + L_pair. Returns dict."""
    l_cls = predicate_loss(pred_logits, target_multihot)
    l_pair = pair_loss(pair_confidence, gt_pairs, pair_pos_weight)
    return {"r_cls": l_cls, "pair": l_pair, "total": l_cls + l_pair}


# =========================================================================== #
# Text-centric scene contrastive loss (eq. 13)
# =========================================================================== #
def text_centric_contrastive_loss(
    text_emb: torch.Tensor,     # (Nt, d) text-modality queries (objects or relations)
    other_emb: torch.Tensor,    # (No, d) other-modality queries
    pos_mask: torch.Tensor,     # (Nt, No) bool: True where the pair is a positive
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE aligning other-modality queries to text queries.

    For each text query with >=1 positive, each positive p gives
        -log( exp(s_ip) / (exp(s_ip) + Σ_{neg} exp(s_in)) ),
    where negatives are the non-positive other-modality queries for that text query
    (matching the denominator in eq. 13). Cosine similarity is used for stability.
    """
    if text_emb.numel() == 0 or other_emb.numel() == 0 or pos_mask.sum() == 0:
        return text_emb.new_zeros(())

    x = F.normalize(text_emb, dim=-1)
    y = F.normalize(other_emb, dim=-1)
    sim = (x @ y.t()) / temperature            # (Nt, No)

    losses = []
    for i in range(sim.size(0)):
        pos_j = pos_mask[i].nonzero(as_tuple=True)[0]
        if pos_j.numel() == 0:
            continue
        neg_j = (~pos_mask[i]).nonzero(as_tuple=True)[0]
        neg_logits = sim[i, neg_j]             # (Nneg,)
        for p in pos_j:
            # log denom = logsumexp([s_ip] + neg_logits)
            cand = torch.cat([sim[i, p].view(1), neg_logits])
            losses.append(-(sim[i, p] - torch.logsumexp(cand, dim=0)))
    if not losses:
        return text_emb.new_zeros(())
    return torch.stack(losses).mean()


def contrastive_loss_total(l_obj_cons: torch.Tensor, l_rel_cons: torch.Tensor) -> torch.Tensor:
    """L_cons = L^o_cons + L^r_cons."""
    return l_obj_cons + l_rel_cons


# =========================================================================== #
# Total loss (eq. 14)
# =========================================================================== #
@dataclass
class LossWeights:
    alpha: float = 1.0   # L_obj
    beta: float = 1.0    # L_ass
    gamma: float = 0.8   # L_rel
    eta: float = 0.6     # L_cons


def total_loss(l_obj, l_ass, l_rel, l_cons, weights: Optional[LossWeights] = None):
    """L = α·L_obj + β·L_ass + γ·L_rel + η·L_cons. Returns (scalar, breakdown dict)."""
    w = weights or LossWeights()
    total = w.alpha * l_obj + w.beta * l_ass + w.gamma * l_rel + w.eta * l_cons
    return total, {"obj": l_obj, "ass": l_ass, "rel": l_rel, "cons": l_cons, "total": total}