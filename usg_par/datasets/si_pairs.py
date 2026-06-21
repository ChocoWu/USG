"""Text-Image (S-I) cross-modal pairs.


"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .psg import PSGDataset

# common caption-word -> PSG-category synonyms (COCO/PSG class names)
SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "people": "person", "guy": "person", "kid": "person", "child": "person",
    "lady": "person", "men": "person", "women": "person", "player": "person",
    "bike": "bicycle", "motorbike": "motorcycle", "television": "tv", "plane": "airplane",
    "cellphone": "cell phone", "couch": "couch", "sofa": "couch", "photo": "picture",
}

# caption-predicate -> PSG predicate synonyms (PSG predicates are mostly spatial)
PRED_SYNONYMS = {
    "next to": "beside", "near": "beside", "by": "beside", "above": "over",
    "below": "under", "underneath": "under", "sitting on": "sitting on",
    "standing on": "standing on", "riding": "riding", "holding": "holding",
    "in front of": "in front of", "on": "on", "in": "in", "over": "over",
    "with": "with", "behind": "behind", "wearing": "wearing", "carrying": "carrying",
}


def parse_caption_triplets(caption: str):
    """caption -> (entity_lemmas: List[str], relations: List[(sub_idx, pred, obj_idx)])."""
    import sng_parser
    g = sng_parser.parse(caption)
    ents = [e["lemma_head"].lower() for e in g["entities"]]
    rels = [(r["subject"], r["relation"].lower(), r["object"]) for r in g["relations"]]
    return ents, rels


def build_category_lookup(object_classes: List[str]) -> Dict[str, int]:
    """PSG class base-name -> class id (e.g. 'building-other-merged' -> 'building')."""
    lookup = {}
    for cid, name in enumerate(object_classes):
        lookup.setdefault(name.split("-")[0].lower(), cid)
        lookup.setdefault(name.lower(), cid)
    return lookup


def build_predicate_lookup(predicate_classes: List[str]) -> Dict[str, int]:
    return {p.lower(): i for i, p in enumerate(predicate_classes)}


def align_entity(lemma: str, lookup: Dict[str, int]) -> int:
    e = SYNONYMS.get(lemma, lemma)
    return lookup.get(e, lookup.get(e.split()[-1], -1) if " " in e else -1)


def align_predicate(pred: str, plookup: Dict[str, int]) -> int:
    p = PRED_SYNONYMS.get(pred, pred)
    return plookup.get(p, -1)


def build_text_sg(caption: str, lookup: Dict[str, int], plookup: Dict[str, int]):
    """caption -> (labels (M,), relations (R,3)) in PSG's class/predicate space.

    Unalignable entities/predicates are dropped; relations referencing a dropped
    entity are removed.
    """
    ents, rels = parse_caption_triplets(caption)
    ent_cls = [align_entity(e, lookup) for e in ents]
    keep = [i for i, c in enumerate(ent_cls) if c >= 0]
    old2new = {i: k for k, i in enumerate(keep)}
    labels = torch.tensor([ent_cls[i] for i in keep], dtype=torch.long)

    out = []
    for s, p, o in rels:
        pid = align_predicate(p, plookup)
        if s in old2new and o in old2new and pid >= 0:
            out.append([old2new[s], old2new[o], pid])
    relations = torch.tensor(out, dtype=torch.long) if out else torch.zeros(0, 3, dtype=torch.long)
    return labels, relations


def dedupe_text_sg(labels: torch.Tensor, relations: torch.Tensor):
    """Collapse duplicate-category text entities to one (long captions repeat nouns).

    Keeps the first occurrence per class; relations are remapped to the kept index and
    self-loops dropped.
    """
    if labels.numel() == 0:
        return labels, relations
    first, remap = {}, {}
    keep = []
    for i, c in enumerate(labels.tolist()):
        if c not in first:
            first[c] = len(keep); keep.append(i)
        remap[i] = first[c]
    new_labels = labels[torch.tensor(keep, dtype=torch.long)]
    rels = [[remap[s], remap[o], p] for s, o, p in relations.tolist() if remap[s] != remap[o]]
    new_rels = torch.tensor(rels, dtype=torch.long) if rels else torch.zeros(0, 3, dtype=torch.long)
    return new_labels, new_rels


def build_category_association(text_labels: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
    """GT association (M_text, M_img): 1 where the two objects share a PSG class."""
    if text_labels.numel() == 0 or image_labels.numel() == 0:
        return torch.zeros(text_labels.numel(), image_labels.numel())
    return (text_labels[:, None] == image_labels[None, :]).float()


class SIPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_json: str, psg_ann, image_root: str, preprocess=None,
                 tokenizer=None, split: str = "train", mask_size=(80, 80)):
        super().__init__()
        with open(pairs_json) as f:
            self.pairs = json.load(f)["pairs"]
        self.psg = PSGDataset(psg_ann, image_root=image_root, preprocess=preprocess,
                              split=split, mask_size=mask_size)
        self.id2idx = {it["image_id"]: i for i, it in enumerate(self.psg.items)}
        self.tokenizer = tokenizer
        self.object_classes = self.psg.object_classes
        self.predicate_classes = self.psg.predicate_classes
        self.lookup = build_category_lookup(self.object_classes)
        self.plookup = build_predicate_lookup(self.predicate_classes)

    @property
    def num_object_classes(self) -> int:
        return self.psg.num_object_classes

    @property
    def num_predicates(self) -> int:
        return self.psg.num_predicates

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Dict:
        pair = self.pairs[i]
        img = self.psg[self.id2idx[pair["image_id"]]]                 # image side (PSG)
        t_labels, t_rel = build_text_sg(pair["caption"], self.lookup, self.plookup)
        tokens = self.tokenizer([pair["caption"]])[0] if self.tokenizer is not None else None
        return {
            "tokens": tokens, "caption": pair["caption"],
            "text_labels": t_labels, "text_relations": t_rel,
            "image": img["image"], "image_labels": img["labels"],
            "image_masks": img["masks"], "image_relations": img["relations"],
            "image_id": pair["image_id"],
        }


def si_collate(batch: List[Dict]) -> Dict:
    tokens = [b["tokens"] for b in batch]
    tokens = torch.stack(tokens) if tokens[0] is not None else None
    images = [b["image"] for b in batch]
    images = torch.stack(images) if images[0] is not None else None
    return {
        "tokens": tokens, "images": images,
        "text_labels": [b["text_labels"] for b in batch],
        "text_relations": [b["text_relations"] for b in batch],
        "image_labels": [b["image_labels"] for b in batch],
        "image_masks": [b["image_masks"] for b in batch] if batch[0]["image_masks"] is not None else None,
        "image_relations": [b["image_relations"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }