"""Relation supervision target construction.

After Hungarian matching assigns object queries to GT entities, we translate the
GT relation triplets (over GT segments) into supervision for:
  * the Pair Confidence Matrix C  -> ``pair_gt`` (N, N), L_pair;
  * the RPC top-k selected pairs   -> ``predicate_target`` (k, P), L^r_cls.
"""

from typing import Tuple

import torch


def matches_to_query2gt(pred_idx: torch.Tensor, gt_idx: torch.Tensor, num_queries: int) -> torch.Tensor:
    """Build a (N,) map query_index -> matched GT segment index (-1 if unmatched)."""
    q2g = torch.full((num_queries,), -1, dtype=torch.long)
    q2g[pred_idx] = gt_idx
    return q2g


def build_association_targets(
    image_match,                  # (pred_idx, gt_idx) for the image modality
    video_match,                  # (pred_idx, gt_idx) for the video modality
    obj_assoc: torch.Tensor,      # (M_img, M_vid) GT object-level association (shared object_id)
    num_queries: int,
) -> torch.Tensor:
    """Query-level association GT (N, N) for the Object Associator (L_ass).

    Maps each modality's query->GT matching through the GT object correspondence:
    A_gt[qi, qj] = 1 iff query qi matched an image object and query qj matched a video
    object that are associated (share an object_id).
    """
    img_q2g = matches_to_query2gt(*image_match, num_queries)   # (N,)
    vid_q2g = matches_to_query2gt(*video_match, num_queries)
    A_gt = torch.zeros(num_queries, num_queries)
    img_q = (img_q2g >= 0).nonzero(as_tuple=True)[0]
    vid_q = (vid_q2g >= 0).nonzero(as_tuple=True)[0]
    if img_q.numel() == 0 or vid_q.numel() == 0 or obj_assoc.numel() == 0:
        return A_gt
    sub = obj_assoc[img_q2g[img_q]][:, vid_q2g[vid_q]]         # (|img_q|, |vid_q|)
    A_gt[img_q[:, None], vid_q[None, :]] = sub.float()
    return A_gt


def build_relation_targets(
    sub_idx: torch.Tensor,        # (k,) subject query indices of RPC-selected pairs
    obj_idx: torch.Tensor,        # (k,) object query indices of RPC-selected pairs
    query2gt: torch.Tensor,       # (N,) query -> GT segment idx (-1 if unmatched)
    gt_relations: torch.Tensor,   # (R, 3) [sub_seg, obj_seg, predicate]
    num_queries: int,
    num_predicates: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (pair_gt (N,N), predicate_target (k, P)) for one sample.

    pair_gt[i,j] = 1 iff query i matched a GT segment that is the subject and query j
    matched the object of some GT relation. predicate_target[t] is the multi-hot of
    predicates for the t-th selected pair (background = all zeros).
    """
    device = sub_idx.device
    pair_gt = torch.zeros(num_queries, num_queries, device=device)
    k = sub_idx.shape[0]
    predicate_target = torch.zeros(k, num_predicates, device=device)

    if gt_relations.numel() == 0:
        return pair_gt, predicate_target

    # invert query2gt: GT segment -> query (a GT seg matches at most one query)
    gt2query = {}
    for q in range(num_queries):
        g = int(query2gt[q])
        if g >= 0:
            gt2query[g] = q

    # (gt_sub_query, gt_obj_query) -> set of predicate ids
    pair_preds = {}
    for s, o, p in gt_relations.tolist():
        qs = gt2query.get(s)
        qo = gt2query.get(o)
        if qs is None or qo is None:   # GT entity not matched to any query
            continue
        pair_gt[qs, qo] = 1.0
        pair_preds.setdefault((qs, qo), set()).add(p)

    # assign predicate multi-hot to the RPC-selected pairs that hit a GT pair
    for t in range(k):
        preds = pair_preds.get((int(sub_idx[t]), int(obj_idx[t])))
        if preds:
            for p in preds:
                predicate_target[t, p] = 1.0
    return pair_gt, predicate_target