"""Image-3D (I-D) cross-modal pairs from 3DSSG/3RScan.

Each pair = a 3D scene (3RScan point cloud) + a 2D image view (a sequence frame of
the same scan). 
A 3D object is "grounded" in the image iff it projects onto the
visible depth surface (scan_projection.visible_object_masks), so the cross-modal
association is free: image object i <-> 3D object j iff they share object_id.

A pairing recipe (scan_id, frame) is stored under data/multimodal/I-D/; the point
cloud, image, masks and association are derived on the fly.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .scan3rscan import read_3rscan_ply, scan_ply_path
from .scan_projection import open_sequence, visible_object_masks
from .threeddsg import build_point_targets, pc_normalize, read_txt_list, subsample_indices


def build_iv_association(image_object_ids: torch.Tensor, scene_object_ids: torch.Tensor) -> torch.Tensor:
    """GT association (M_img, M_3d): 1 where the two objects share an object_id."""
    if image_object_ids.numel() == 0 or scene_object_ids.numel() == 0:
        return torch.zeros(image_object_ids.numel(), scene_object_ids.numel())
    return (image_object_ids[:, None] == scene_object_ids[None, :]).float()


# --------------------------------------------------------------------------- #
# recipe construction (offline; projects candidate frames to pick good views)
# --------------------------------------------------------------------------- #
def build_id_pairs_for_scan(pc, scan_dir, frames_per_scan=3, candidates=10, min_objects=3):
    """Pick the frames with the most grounded objects for one scan."""
    try:
        zf, info = open_sequence(scan_dir)
    except Exception:
        return []
    cand = np.linspace(0, info.num_frames - 1, candidates).astype(int)
    scored = []
    for fi in cand:
        frame = f"frame-{int(fi):06d}"
        try:
            ids, _ = visible_object_masks(pc, zf, frame, info, min_points=30)
        except Exception:
            continue
        if len(ids) >= min_objects:
            scored.append((len(ids), frame))
    scored.sort(reverse=True)
    return [{"frame": f, "num_grounded": int(n)} for n, f in scored[:frames_per_scan]]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class IDPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_json: str, rscan_root: str, ssg_root: str,
                 preprocess=None, num_points: int = 4096, mask_size=(80, 80), normalize=True):
        super().__init__()
        with open(pairs_json) as f:
            self.pairs = json.load(f)["pairs"]
        self.rscan_root = rscan_root
        self.preprocess = preprocess
        self.num_points = num_points
        self.mask_size = tuple(mask_size)
        self.normalize = normalize

        self.object_classes = [l.split("\t")[1] for l in read_txt_list(os.path.join(ssg_root, "classes.txt"))]
        self.predicate_classes = read_txt_list(os.path.join(ssg_root, "relationships.txt"))
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
        sid, frame = pair["scan_id"], pair["frame"]
        pc = read_3rscan_ply(scan_ply_path(self.rscan_root, sid))

        # 3D modality: subsample -> points + targets (+ object_ids)
        idx = subsample_indices(pc.num_points, self.num_points)
        xyz = pc_normalize(pc.xyz[idx]) if self.normalize else pc.xyz[idx]
        points = torch.from_numpy(np.concatenate([xyz, pc.rgb[idx]], axis=1)).float()
        p_lab, p_msk, p_rel, p_ids = build_point_targets(
            pc.object_id[idx], self._obj_gid.get(sid, {}), self._rels.get(sid, []), return_ids=True)

        # image modality: color frame + grounded objects (full-resolution point cloud for visibility)
        zf, info = open_sequence(os.path.join(self.rscan_root, sid))
        img_ids, img_masks = visible_object_masks(pc, zf, frame, info, mask_size=self.mask_size)
        gid = self._obj_gid.get(sid, {})
        keep = [k for k, oid in enumerate(img_ids.tolist()) if oid in gid]
        img_ids = img_ids[keep]
        img_masks = img_masks[keep]
        img_labels = torch.tensor([gid[int(o)] - 1 for o in img_ids.tolist()], dtype=torch.long)
        # image relations: scene relations among grounded objects
        local = {int(o): k for k, o in enumerate(img_ids.tolist())}
        irel = [[local[s], local[o], p - 1] for s, o, p in self._rels.get(sid, [])
                if s in local and o in local]
        img_relations = torch.tensor(irel, dtype=torch.long) if irel else torch.zeros(0, 3, dtype=torch.long)

        from .scan_projection import read_color
        image = self.preprocess(read_color(zf, frame)) if self.preprocess is not None else None

        return {
            "image": image, "image_labels": img_labels, "image_masks": img_masks,
            "image_relations": img_relations, "image_object_ids": img_ids,
            "points": points, "point_labels": p_lab, "point_masks": p_msk,
            "point_relations": p_rel, "point_object_ids": p_ids,
            "scan_id": sid, "frame": frame,
        }


def id_collate(batch: List[Dict]) -> Dict:
    """Collate I-D pairs (both modalities batch B; no frame folding)."""
    images = torch.stack([b["image"] for b in batch]) if batch[0]["image"] is not None else None
    return {
        "images": images,
        "points": torch.stack([b["points"] for b in batch]),
        "image_labels": [b["image_labels"] for b in batch],
        "image_masks": [b["image_masks"] for b in batch],
        "image_relations": [b["image_relations"] for b in batch],
        "image_object_ids": [b["image_object_ids"] for b in batch],
        "point_labels": [b["point_labels"] for b in batch],
        "point_masks": [b["point_masks"] for b in batch],
        "point_relations": [b["point_relations"] for b in batch],
        "point_object_ids": [b["point_object_ids"] for b in batch],
        "scan_ids": [b["scan_id"] for b in batch],
    }