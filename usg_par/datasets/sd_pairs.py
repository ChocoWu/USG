"""Text-3D (S-D) cross-modal pairs from 3RScan scene captions + 3DSSG scene graph.

3D side = a 3RScan point cloud + 3DSSG objects/relations (same base as 3DDSG/I-D).
Text side = a natural-language scene caption (data/multimodal/3RScan/.../scene_cap.json,
GPT-generated) parsed into a scene graph whose entities are aligned to 3DSSG's 528
object classes. Cross-modal association is category-level (text entity <-> 3D object
of the same class), analogous to S-I.

Note: the paper's S-D uses ScanRefer/ScanNet; here we have captions for 3RScan
directly, so S-D shares the same 3D base as 3DDSG (no ScanNet mismatch).
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch

from .scan3rscan import read_3rscan_ply, scan_ply_path
from .si_pairs import (
    build_category_lookup,
    build_predicate_lookup,
    build_text_sg,
    dedupe_text_sg,
)
from .threeddsg import build_point_targets, pc_normalize, read_txt_list, subsample_indices


class SDPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_json: str, rscan_root: str, ssg_root: str, tokenizer=None,
                 num_points: int = 4096, mask_size=(80, 80), normalize: bool = True):
        super().__init__()
        with open(pairs_json) as f:
            self.pairs = json.load(f)["pairs"]
        self.rscan_root = rscan_root
        self.tokenizer = tokenizer
        self.num_points = num_points
        self.mask_size = tuple(mask_size)
        self.normalize = normalize

        self.object_classes = [l.split("\t")[1] for l in read_txt_list(os.path.join(ssg_root, "classes.txt"))]
        self.predicate_classes = read_txt_list(os.path.join(ssg_root, "relationships.txt"))
        self.lookup = build_category_lookup(self.object_classes)
        self.plookup = build_predicate_lookup(self.predicate_classes)
        objs = json.load(open(os.path.join(ssg_root, "objects.json")))["scans"]
        rels = json.load(open(os.path.join(ssg_root, "relationships.json")))["scans"]
        self._obj_gid = {s["scan"]: {int(o["id"]): int(o["global_id"]) for o in s["objects"]} for s in objs}
        self._rels = {s["scan"]: [(int(r[0]), int(r[1]), int(r[2])) for r in s["relationships"]] for s in rels}

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Dict:
        pair = self.pairs[i]
        sid, caption = pair["scan_id"], pair["caption"]

        # 3D side
        pc = read_3rscan_ply(scan_ply_path(self.rscan_root, sid))
        idx = subsample_indices(pc.num_points, self.num_points)
        xyz = pc_normalize(pc.xyz[idx]) if self.normalize else pc.xyz[idx]
        points = torch.from_numpy(np.concatenate([xyz, pc.rgb[idx]], axis=1)).float()
        p_lab, p_msk, p_rel = build_point_targets(
            pc.object_id[idx], self._obj_gid.get(sid, {}), self._rels.get(sid, []))

        # text side (dedupe duplicate-category entities in long scene captions)
        t_lab, t_rel = dedupe_text_sg(*build_text_sg(caption, self.lookup, self.plookup))
        tokens = self.tokenizer([caption])[0] if self.tokenizer is not None else None

        return {
            "tokens": tokens, "caption": caption,
            "text_labels": t_lab, "text_relations": t_rel,
            "points": points, "point_labels": p_lab, "point_masks": p_msk,
            "point_relations": p_rel, "scan_id": sid,
        }


def sd_collate(batch: List[Dict]) -> Dict:
    tokens = [b["tokens"] for b in batch]
    tokens = torch.stack(tokens) if tokens[0] is not None else None
    return {
        "tokens": tokens,
        "points": torch.stack([b["points"] for b in batch]),
        "text_labels": [b["text_labels"] for b in batch],
        "text_relations": [b["text_relations"] for b in batch],
        "point_labels": [b["point_labels"] for b in batch],
        "point_masks": [b["point_masks"] for b in batch],
        "point_relations": [b["point_relations"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "scan_ids": [b["scan_id"] for b in batch],
    }