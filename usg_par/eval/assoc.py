"""Association Accuracy@K for cross-modal object alignment.

For each object in modality A that has a GT cross-modal counterpart in B, check
whether the counterpart is among the top-K highest-scored associations predicted by
the Object Associator. 
Acc@K = fraction of such A-objects ranked correctly.

Works at the query level: A_pred / A_gt are (N, N) over the two modalities' queries
(A_gt comes from build_association_targets — the GT object_id correspondence mapped
through each modality's Hungarian matching).
"""

from typing import Sequence

import torch


def assoc_accuracy(a_pred: torch.Tensor, a_gt: torch.Tensor, k: int) -> tuple:
    """Return (correct, total) for one (N,N) association matrix.

    A "query" (row) counts only if it has >=1 GT counterpart; it's correct if the
    top-k predicted columns include a GT-positive column.
    """
    rows = (a_gt.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return 0, 0
    kk = min(k, a_pred.shape[1])
    correct = 0
    for i in rows.tolist():
        topk = a_pred[i].topk(kk).indices
        if (a_gt[i, topk] > 0).any():
            correct += 1
    return correct, rows.numel()


class AssocAccEvaluator:
    """Accumulate Association Accuracy@K over a dataset."""

    def __init__(self, k_list: Sequence[int] = (5,)):
        self.k_list = tuple(k_list)
        self.correct = {k: 0 for k in self.k_list}
        self.total = 0

    def update(self, a_pred: torch.Tensor, a_gt: torch.Tensor):
        # total is counted once (same set of rows for every k)
        _, t = assoc_accuracy(a_pred, a_gt, self.k_list[0])
        self.total += t
        for k in self.k_list:
            c, _ = assoc_accuracy(a_pred, a_gt, k)
            self.correct[k] += c

    def compute(self) -> dict:
        return {f"Acc@{k}": (self.correct[k] / self.total if self.total else 0.0)
                for k in self.k_list}