"""Assemble training losses from model output + GT.

Single modality (PSG/PVSG/FACTUAL/3DDSG):
  L = α·L_obj + γ·L_rel
Multimodal I-V:
  L = α·(L_obj^I + L_obj^V) + β·L_ass + γ·(L_rel^I + L_rel^V)
with the cross-modal association supervised via the GT object_id correspondence.
"""

from typing import Dict, List, Optional

import torch

from ..losses import (
    DetectionLossWeights,
    HungarianMatcher,
    LossWeights,
    association_loss,
    object_detection_loss,
    pair_loss,
    predicate_loss,
    text_centric_contrastive_loss,
)
from ..model import USGOutput
from .targets import build_association_targets, build_relation_targets, matches_to_query2gt


def _modality_terms(mo, gt_labels, gt_masks, gt_relations, num_predicates,
                    matcher: HungarianMatcher, det_weights: Optional[DetectionLossWeights]):
    """Per-modality L_obj, L_rel, and the Hungarian matches. Returns a dict."""
    b, n = mo.cls_logits.shape[:2]
    device = mo.cls_logits.device
    det = object_detection_loss(mo.cls_logits, mo.pred_masks, gt_labels, gt_masks,
                                matcher, det_weights, return_matches=True)
    sub_idx, obj_idx = mo.rpc_out.sub_idx, mo.rpc_out.obj_idx
    pair_gts, pred_targets = [], []
    for bi in range(b):
        pred_i, gt_i = det["matches"][bi]
        q2g = matches_to_query2gt(pred_i, gt_i, n).to(device)
        pair_gt, pred_t = build_relation_targets(
            sub_idx[bi], obj_idx[bi], q2g, gt_relations[bi], n, num_predicates)
        pair_gts.append(pair_gt); pred_targets.append(pred_t)
    l_pair = pair_loss(mo.rpc_out.pair_confidence, torch.stack(pair_gts).to(device))
    l_rcls = predicate_loss(mo.relation_logits, torch.stack(pred_targets).to(device))
    return {"l_obj": det["total"], "l_rel": l_rcls + l_pair, "matches": det["matches"],
            "cls": det["cls"], "ce": det["ce"], "dice": det["dice"], "r_cls": l_rcls, "pair": l_pair}


def compute_single_modality_losses(
    out: USGOutput, modality: str,
    gt_labels: List[torch.Tensor], gt_masks: Optional[List[torch.Tensor]],
    gt_relations: List[torch.Tensor], num_predicates: int,
    matcher: Optional[HungarianMatcher] = None,
    det_weights: Optional[DetectionLossWeights] = None,
    loss_weights: Optional[LossWeights] = None,
):
    """Returns (total_loss, breakdown). L = α·L_obj + γ·L_rel."""
    matcher = matcher or HungarianMatcher()
    lw = loss_weights or LossWeights()
    t = _modality_terms(out.per_modality[modality], gt_labels, gt_masks, gt_relations,
                        num_predicates, matcher, det_weights)
    total = lw.alpha * t["l_obj"] + lw.gamma * t["l_rel"]
    breakdown = {"total": total, "obj": t["l_obj"], "rel": t["l_rel"],
                 "cls": t["cls"], "ce": t["ce"], "dice": t["dice"],
                 "r_cls": t["r_cls"], "pair": t["pair"]}
    return total, breakdown


def compute_multimodal_losses(
    out: USGOutput,
    gt: Dict[str, Dict],                      # modality -> {labels, masks, relations}
    obj_assoc: List[torch.Tensor],            # length B*: GT object association (M_a, M_b) per sample
    pair: tuple = ("image", "video"),
    num_predicates_per_modality: Optional[Dict[str, int]] = None,
    matcher: Optional[HungarianMatcher] = None,
    loss_weights: Optional[LossWeights] = None,
):
    """L = α·(L_obj^A + L_obj^B) + β·L_ass + γ·(L_rel^A + L_rel^B). Returns (total, breakdown)."""
    matcher = matcher or HungarianMatcher()
    lw = loss_weights or LossWeights()
    a, b = pair
    npp = num_predicates_per_modality or {}

    terms = {}
    for m in (a, b):
        mo = out.per_modality[m]
        terms[m] = _modality_terms(
            mo, gt[m]["labels"], gt[m]["masks"], gt[m]["relations"],
            npp.get(m, mo.relation_logits.shape[-1]), matcher, None)

    # association loss over the cross-modal pair (L_ass)
    key = "|".join(sorted(pair))
    a_pred = out.associations[key]                         # (B*, N, N) logits; rows = sorted-first modality
    src, tgt = sorted(pair)                                # associator orients rows to `src`
    n = a_pred.shape[1]
    a_gts = []
    for k in range(a_pred.shape[0]):
        oa = obj_assoc[k]                                  # (M_a, M_b) for the original pair order (a,b)
        oa = oa if src == a else oa.t()                    # orient to (src, tgt)
        a_gts.append(build_association_targets(
            terms[src]["matches"][k], terms[tgt]["matches"][k], oa, n))
    l_ass = association_loss(a_pred, torch.stack(a_gts).to(a_pred.device))

    # text-centric scene contrastive loss: only when text is one modality.
    # Align the visual/3D modality's object queries to the text queries; positives are
    # the corresponding (associated) query pairs (reuse the query-level association GT).
    l_cons = a_pred.new_zeros(())
    if "text" in pair:
        other = b if a == "text" else a
        text_q = out.per_modality["text"].refined_query
        other_q = out.per_modality[other].refined_query
        for k in range(a_pred.shape[0]):
            # a_gts[k] is (src, tgt) oriented; want pos_mask (N_text, N_other)
            pos = (a_gts[k] if src == "text" else a_gts[k].t()) > 0
            l_cons = l_cons + text_centric_contrastive_loss(
                text_q[k], other_q[k], pos.to(text_q.device))
        l_cons = l_cons / a_pred.shape[0]

    l_obj = terms[a]["l_obj"] + terms[b]["l_obj"]
    l_rel = terms[a]["l_rel"] + terms[b]["l_rel"]
    total = lw.alpha * l_obj + lw.beta * l_ass + lw.gamma * l_rel + lw.eta * l_cons
    breakdown = {"total": total, "obj": l_obj, "ass": l_ass, "rel": l_rel, "cons": l_cons,
                 f"obj_{a}": terms[a]["l_obj"], f"obj_{b}": terms[b]["l_obj"]}
    return total, breakdown