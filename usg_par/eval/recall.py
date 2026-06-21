"""Recall@K and mean Recall@K for scene graph detection.

A predicted triplet (sub_label, predicate, obj_label) with subject/object masks is a
hit for a GT triplet iff the three labels match AND both subject and object mask IoUs
exceed a threshold (default 0.5). 
Each GT triplet may be matched at most once; greedy
matching in descending predicted-score order. 
R@K uses the top-K predicted triplets;
mR@K averages per-predicate recall.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Triplet:
    sub_label: int
    predicate: int
    obj_label: int
    score: float = 1.0
    sub_mask: Optional[torch.Tensor] = None   # (H, W) binary
    obj_mask: Optional[torch.Tensor] = None


def mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU between two binary masks (same shape)."""
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union > 0 else 0.0


def _triplet_matches(pt: Triplet, gt: Triplet, iou_thr: float, use_mask: bool) -> bool:
    if not (pt.sub_label == gt.sub_label and pt.obj_label == gt.obj_label
            and pt.predicate == gt.predicate):
        return False
    if not use_mask:
        return True
    return (mask_iou(pt.sub_mask, gt.sub_mask) >= iou_thr
            and mask_iou(pt.obj_mask, gt.obj_mask) >= iou_thr)


class SGRecallEvaluator:
    """Accumulates R@K and mR@K over a dataset."""

    def __init__(self, k_list=(20, 50, 100), num_predicates: int = 56,
                 iou_thr: float = 0.5, use_mask: bool = True):
        self.k_list = tuple(k_list)
        self.num_predicates = num_predicates
        self.iou_thr = iou_thr
        self.use_mask = use_mask
        # R@K accumulators
        self.hits = {k: 0 for k in self.k_list}
        self.total = 0
        # mR@K accumulators (per predicate class)
        self.cls_hits = {k: [0] * num_predicates for k in self.k_list}
        self.cls_total = [0] * num_predicates

    def update(self, pred_triplets: List[Triplet], gt_triplets: List[Triplet]):
        """Add one image's predictions vs GT."""
        self.total += len(gt_triplets)
        for gt in gt_triplets:
            self.cls_total[gt.predicate] += 1
        if not gt_triplets:
            return

        preds = sorted(pred_triplets, key=lambda t: t.score, reverse=True)
        # greedy match: record the pred rank at which each GT is first matched
        gt_match_rank = [None] * len(gt_triplets)
        used = [False] * len(gt_triplets)
        for rank, pt in enumerate(preds):
            for gi, gt in enumerate(gt_triplets):
                if used[gi]:
                    continue
                if _triplet_matches(pt, gt, self.iou_thr, self.use_mask):
                    used[gi] = True
                    gt_match_rank[gi] = rank
                    break

        for k in self.k_list:
            for gi, gt in enumerate(gt_triplets):
                if gt_match_rank[gi] is not None and gt_match_rank[gi] < k:
                    self.hits[k] += 1
                    self.cls_hits[k][gt.predicate] += 1

    def compute(self) -> dict:
        out = {}
        for k in self.k_list:
            out[f"R@{k}"] = self.hits[k] / self.total if self.total else 0.0
            per_cls = [self.cls_hits[k][c] / self.cls_total[c]
                       for c in range(self.num_predicates) if self.cls_total[c] > 0]
            out[f"mR@{k}"] = sum(per_cls) / len(per_cls) if per_cls else 0.0
        return out