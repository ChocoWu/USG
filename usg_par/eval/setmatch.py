"""Set Match metric for text scene-graph parsing.

Set Match = fraction of captions whose predicted triplet SET exactly equals the GT triplet set (order-independent, normalized by lowercasing/stripping).
"""

from typing import List, Optional, Tuple

import torch


def _norm(triplets: List[Tuple[str, str, str]]):
    return {tuple(x.strip().lower() for x in t) for t in triplets}


def set_match(pred_triplets: List[Tuple[str, str, str]],
              gt_triplets: List[Tuple[str, str, str]]) -> float:
    """1.0 if the predicted triplet set exactly equals the GT set, else 0.0."""
    return 1.0 if _norm(pred_triplets) == _norm(gt_triplets) else 0.0


class SetMatchEvaluator:
    """Accumulates mean Set Match over a dataset.

    Captions whose GT parses to an empty triplet set are skipped: every FACTUAL
    caption has a scene graph, so an empty GT is a parse failure and an empty==empty
    "match" would spuriously inflate the score.
    """

    def __init__(self):
        self.hits = 0.0
        self.total = 0
        self.skipped = 0

    def update(self, pred_triplets, gt_triplets):
        if not gt_triplets:
            self.skipped += 1
            return
        self.hits += set_match(pred_triplets, gt_triplets)
        self.total += 1

    def compute(self) -> dict:
        return {"SetMatch": self.hits / self.total if self.total else 0.0}


@torch.no_grad()
def build_text_pred_triplets(
    cls_logits: torch.Tensor,        # (N, C+1)
    sub_idx: torch.Tensor,           # (k,)
    obj_idx: torch.Tensor,           # (k,)
    relation_logits: torch.Tensor,   # (k, P)
    object_classes: List[str],
    predicate_classes: List[str],
    pair_scores: Optional[torch.Tensor] = None,   # (k,) RPC confidence
    score_thr: float = 0.5,
) -> List[Tuple[str, str, str]]:
    """Compose a predicted name-triplet set for one caption.

    A pair contributes a triplet if its predicate's max sigmoid prob exceeds
    ``score_thr`` (and, if given, its pair confidence is positive). Object/predicate
    labels are argmaxes mapped back to names. Returns a de-duplicated list.
    """
    num_obj = len(object_classes)
    obj_label = cls_logits[:, :num_obj].argmax(-1)        # exclude no-object col
    pred_prob = relation_logits.sigmoid()
    triplets = set()
    for t in range(sub_idx.shape[0]):
        p_score, p = float(pred_prob[t].max()), int(pred_prob[t].argmax())
        if p_score < score_thr:
            continue
        if pair_scores is not None and float(pair_scores[t]) <= 0:
            continue
        si, oi = int(sub_idx[t]), int(obj_idx[t])
        triplets.add((object_classes[int(obj_label[si])], predicate_classes[p],
                      object_classes[int(obj_label[oi])]))
    return list(triplets)