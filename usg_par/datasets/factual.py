"""FACTUAL text scene-graph dataset.

CSV columns: image_id, region_id, caption, scene_graph. 
The scene_graph string is a list of triplets ``( subject , predicate , object )`` joined by ` , `; attributes are encoded as ``( entity , is , attribute )``. 

"""

import csv
import re
from typing import Dict, List, Optional, Tuple

import torch

_TRIPLET_RE = re.compile(r"\(([^()]*)\)")


def parse_scene_graph(s: str) -> List[Tuple[str, str, str]]:
    """Parse a FACTUAL scene_graph string into (subject, predicate, object) tuples."""
    out = []
    for grp in _TRIPLET_RE.findall(s):
        parts = [p.strip() for p in grp.split(",")]
        if len(parts) == 3 and all(parts):
            out.append((parts[0], parts[1], parts[2]))
    return out


def build_factual_vocab(csv_path: str) -> Tuple[List[str], List[str]]:
    """Build (object_classes, predicate_classes) from a FACTUAL CSV (e.g. train split)."""
    objs, preds = set(), set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for s, p, o in parse_scene_graph(row["scene_graph"]):
                objs.add(s); objs.add(o); preds.add(p)
    return sorted(objs), sorted(preds)


class FACTUALDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer=None,
        object_classes: Optional[List[str]] = None,
        predicate_classes: Optional[List[str]] = None,
        max_objects: int = 100,
    ):
        super().__init__()
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))
        # vocab: provided (e.g. train's, for dev/test) or built from this file
        if object_classes is None or predicate_classes is None:
            object_classes, predicate_classes = build_factual_vocab(csv_path)
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.obj_to_id = {n: i for i, n in enumerate(object_classes)}
        self.pred_to_id = {n: i for i, n in enumerate(predicate_classes)}
        self.tokenizer = tokenizer
        self.max_objects = max_objects

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict:
        row = self.rows[i]
        triplets = parse_scene_graph(row["scene_graph"])

        # unique entities -> object labels; relations over local indices
        entities: List[str] = []
        for s, _, o in triplets:
            for e in (s, o):
                if e not in entities:
                    entities.append(e)
        ent_local = {e: k for k, e in enumerate(entities)}

        labels = torch.tensor([self.obj_to_id.get(e, -1) for e in entities], dtype=torch.long)
        rels = []
        for s, p, o in triplets:
            pid = self.pred_to_id.get(p, -1)
            if pid >= 0 and labels[ent_local[s]] >= 0 and labels[ent_local[o]] >= 0:
                rels.append([ent_local[s], ent_local[o], pid])
        relations = torch.tensor(rels, dtype=torch.long) if rels else torch.zeros(0, 3, dtype=torch.long)

        tokens = self.tokenizer([row["caption"]])[0] if self.tokenizer is not None else None
        return {
            "tokens": tokens,
            "caption": row["caption"],
            "labels": labels,
            "masks": None,                       # text SG has no masks
            "relations": relations,
            "gt_triplets": triplets,             # raw names, for Set-Match eval
        }


def factual_collate(batch: List[Dict]) -> Dict:
    tokens = [b["tokens"] for b in batch]
    tokens = torch.stack(tokens) if tokens[0] is not None else None
    return {
        "tokens": tokens,
        "captions": [b["caption"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "masks": None,
        "relations": [b["relations"] for b in batch],
        "gt_triplets": [b["gt_triplets"] for b in batch],
    }