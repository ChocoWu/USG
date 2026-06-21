"""Unified (union) vocabulary across datasets for joint multi-modal training.

Stage-1 joint training interleaves the four single-modality datasets (PSG image,
PVSG video, FACTUAL text, 3DDSG point). Their object/predicate label spaces differ,
so we build a single **union vocabulary** (classes merged by name — identical names
across datasets share one id) plus per-source remap tables, and remap each sample's
labels/relations to the union space at load time. Open-vocab cosine classification
means the union just sizes the class-name embedding set; the relation head is a
Linear over the union predicate count.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class UnifiedVocab:
    object_classes: List[str]                       # union object class names
    predicate_classes: List[str]                    # union predicate names
    obj_remap: Dict[str, List[int]] = field(default_factory=dict)   # source -> [local_id -> union_id]
    pred_remap: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)


def _union(name_lists: Dict[str, Sequence[str]]) -> Tuple[List[str], Dict[str, List[int]]]:
    union, index, remap = [], {}, {}
    for src, names in name_lists.items():
        r = []
        for n in names:
            if n not in index:
                index[n] = len(union)
                union.append(n)
            r.append(index[n])
        remap[src] = r
    return union, remap


def build_unified_vocab(object_class_lists: Dict[str, Sequence[str]],
                        predicate_class_lists: Dict[str, Sequence[str]]) -> UnifiedVocab:
    """Build the union object/predicate vocab + per-source local->union remaps."""
    obj_union, obj_remap = _union(object_class_lists)
    pred_union, pred_remap = _union(predicate_class_lists)
    return UnifiedVocab(obj_union, pred_union, obj_remap, pred_remap)


class RemapWrapper(torch.utils.data.Dataset):
    """Wrap a dataset to remap its object labels + relation predicates to union ids.

    Args:
        dataset: the base dataset (single-modal or a pair dataset).
        obj_remap, pred_remap: local_id -> union_id lists (from the dataset's source).
        label_fields: dict keys holding object-label tensors to remap.
        relation_fields: dict keys holding (R,3) relation tensors (col 2 = predicate).
    """

    def __init__(self, dataset, obj_remap: List[int], pred_remap: List[int],
                 label_fields=("labels",), relation_fields=("relations",)):
        self.dataset = dataset
        self.obj_remap = torch.tensor(obj_remap, dtype=torch.long)
        self.pred_remap = torch.tensor(pred_remap, dtype=torch.long)
        self.label_fields = tuple(label_fields)
        self.relation_fields = tuple(relation_fields)

    def __len__(self) -> int:
        return len(self.dataset)

    def _remap_obj(self, t):
        return self.obj_remap[t] if t.numel() > 0 else t

    def _remap_rel(self, r):
        if r.numel() > 0:
            r = r.clone()
            r[:, 2] = self.pred_remap[r[:, 2]]
        return r

    def __getitem__(self, i: int) -> Dict:
        # fields may be a tensor or a list of tensors (e.g. video per-frame labels)
        item = dict(self.dataset[i])
        for f in self.label_fields:
            v = item.get(f)
            if v is None:
                continue
            item[f] = [self._remap_obj(t) for t in v] if isinstance(v, list) else self._remap_obj(v)
        for f in self.relation_fields:
            v = item.get(f)
            if v is None:
                continue
            item[f] = [self._remap_rel(r) for r in v] if isinstance(v, list) else self._remap_rel(v)
        return item