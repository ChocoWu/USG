"""Build predicted / GT triplets for scene-graph recall evaluation."""

from typing import List, Optional

import torch

from .recall import Triplet


@torch.no_grad()
def build_pred_triplets(
    cls_logits: torch.Tensor,        # (N, C+1) for one image
    pred_masks: Optional[torch.Tensor],  # (N, H, W) or None
    sub_idx: torch.Tensor,           # (k,) subject query indices
    obj_idx: torch.Tensor,           # (k,)
    relation_logits: torch.Tensor,   # (k, P)
    mask_thr: float = 0.5,
    top_predicates: int = 1,         # triplets generated per pair (top predicates)
) -> List[Triplet]:
    """Compose ranked predicted triplets for one image.

    Object label = argmax over the C real classes (excludes the no-object column);
    object score = its softmax prob. Triplet score = sub_score * obj_score * pred_score.
    """
    num_obj_cls = cls_logits.shape[-1] - 1
    obj_prob = cls_logits[:, :num_obj_cls].softmax(-1)        # (N, C)
    obj_score, obj_label = obj_prob.max(-1)                   # (N,)
    bin_masks = (pred_masks > mask_thr) if pred_masks is not None else None
    pred_prob = relation_logits.sigmoid()                    # (k, P)

    triplets: List[Triplet] = []
    k = sub_idx.shape[0]
    top_predicates = min(top_predicates, pred_prob.shape[-1])
    for t in range(k):
        si, oi = int(sub_idx[t]), int(obj_idx[t])
        s_score = float(obj_score[si]); o_score = float(obj_score[oi])
        top_p = pred_prob[t].topk(top_predicates).indices.tolist()
        for p in top_p:
            score = s_score * o_score * float(pred_prob[t, p])
            triplets.append(Triplet(
                sub_label=int(obj_label[si]), predicate=int(p), obj_label=int(obj_label[oi]),
                score=score,
                sub_mask=bin_masks[si] if bin_masks is not None else None,
                obj_mask=bin_masks[oi] if bin_masks is not None else None,
            ))
    return triplets


def build_gt_triplets(
    labels: torch.Tensor,            # (M,) object labels for one image
    masks: Optional[torch.Tensor],   # (M, H, W) or None
    relations: torch.Tensor,         # (R, 3) [sub_seg, obj_seg, predicate]
) -> List[Triplet]:
    """Compose GT triplets for one image."""
    triplets: List[Triplet] = []
    for s, o, p in relations.tolist():
        triplets.append(Triplet(
            sub_label=int(labels[s]), predicate=int(p), obj_label=int(labels[o]),
            sub_mask=masks[s] if masks is not None else None,
            obj_mask=masks[o] if masks is not None else None,
        ))
    return triplets