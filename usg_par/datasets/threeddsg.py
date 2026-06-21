"""3DDSG (3DSSG-on-3RScan) point-cloud scene-graph dataset (paper Table 4).

Per scan we read the annotated point cloud (xyz+rgb + per-point instance/class ids,
see scan3rscan.py), subsample to a fixed number of points, and build:
  points     (N, 6)   xyz (normalized) + rgb           -> point-cloud input D
  gt_labels  (M,)     object class ids (global_id - 1)  -> open-vocab via class names
  gt_masks   (M, N)   per-point instance masks          -> 3D instance masks mi
  relations  (R, 3)   (sub_local, obj_local, pred_id-1) -> intra-3D triplets

Object/relation vocab come from 3DSSG classes.txt (527) / relationships.txt (40,
"none" + 39 predicates). This is the GT setup (GT instance segmentation), matching
the SGCls/PreCls tasks in Table 4.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .scan3rscan import read_3rscan_ply, scan_ply_path


def read_txt_list(path: str) -> List[str]:
    return [l for l in open(path).read().splitlines() if l.strip()]


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    """Center to centroid and scale to the unit sphere (Point-BERT convention)."""
    xyz = xyz - xyz.mean(0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    return xyz / (scale + 1e-6)


def subsample_indices(p: int, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Pick n point indices from p (without replacement if p>=n, else pad with repeats)."""
    rng = rng or np.random.default_rng()
    if p >= n:
        return rng.choice(p, n, replace=False)
    return np.concatenate([np.arange(p), rng.choice(p, n - p, replace=True)])


def build_point_targets(
    object_id: np.ndarray,                 # (N,) per-point instance id
    obj_global_id: Dict[int, int],         # objectId -> class global_id (1..527)
    relations: List[Tuple[int, int, int]], # (sub_objectId, obj_objectId, pred_id 1-indexed)
    return_ids: bool = False,
):
    """Build (gt_labels (M,), gt_masks (M,N), relations (R,3)) for one scan.

    Objects are those present in the (subsampled) point cloud that also have a known
    class; relations with an absent endpoint are dropped. Labels are global_id-1,
    predicate ids are pred_id-1 (0-indexed; 0 == "none"). If ``return_ids``, also
    returns the present object_ids (M,) (for cross-modal association).
    """
    present = [int(i) for i in np.unique(object_id) if i != 0 and int(i) in obj_global_id]
    local = {oid: k for k, oid in enumerate(present)}
    labels = torch.tensor([obj_global_id[oid] - 1 for oid in present], dtype=torch.long)
    if present:
        masks = torch.from_numpy(
            np.stack([(object_id == oid) for oid in present]).astype(np.float32))
    else:
        masks = torch.zeros(0, object_id.shape[0])

    rels = []
    for sub, obj, pid in relations:
        if sub in local and obj in local:
            rels.append([local[sub], local[obj], pid - 1])
    relations_t = torch.tensor(rels, dtype=torch.long) if rels else torch.zeros(0, 3, dtype=torch.long)
    if return_ids:
        return labels, masks, relations_t, torch.tensor(present, dtype=torch.long)
    return labels, masks, relations_t


class Scan3DSSGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rscan_root: str,                   # data/3DSG/3RScan
        ssg_root: str,                     # data/3DSG/3DSSG
        split_file: str,                   # data/3DSG/split/{train,validation,test}_scans.txt
        num_points: int = 4096,
        normalize: bool = True,
    ):
        super().__init__()
        self.rscan_root = rscan_root
        self.num_points = num_points
        self.normalize = normalize

        # vocab
        self.object_classes = [l.split("\t")[1] for l in read_txt_list(os.path.join(ssg_root, "classes.txt"))]
        self.predicate_classes = read_txt_list(os.path.join(ssg_root, "relationships.txt"))

        # per-scan objects (id->global_id) and relationships
        objs = json.load(open(os.path.join(ssg_root, "objects.json")))["scans"]
        rels = json.load(open(os.path.join(ssg_root, "relationships.json")))["scans"]
        self._obj_gid = {s["scan"]: {int(o["id"]): int(o["global_id"]) for o in s["objects"]} for s in objs}
        self._rels = {s["scan"]: [(int(r[0]), int(r[1]), int(r[2])) for r in s["relationships"]] for s in rels}

        # keep scans in this split that are downloaded AND have SG annotations
        wanted = read_txt_list(split_file)
        self.scans = [s for s in wanted
                      if s in self._rels and os.path.isfile(scan_ply_path(rscan_root, s))]

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, i: int) -> Dict:
        sid = self.scans[i]
        pc = read_3rscan_ply(scan_ply_path(self.rscan_root, sid))
        idx = subsample_indices(pc.num_points, self.num_points)
        xyz = pc_normalize(pc.xyz[idx]) if self.normalize else pc.xyz[idx]
        points = torch.from_numpy(np.concatenate([xyz, pc.rgb[idx]], axis=1)).float()  # (N,6)
        object_id = pc.object_id[idx]

        labels, masks, relations = build_point_targets(
            object_id, self._obj_gid.get(sid, {}), self._rels.get(sid, []))
        return {
            "points": points, "labels": labels, "masks": masks,
            "relations": relations, "scan_id": sid,
        }


def threeddsg_collate(batch: List[Dict]) -> Dict:
    return {
        "points": torch.stack([b["points"] for b in batch]),     # (B, N, 6)
        "labels": [b["labels"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "relations": [b["relations"] for b in batch],
        "scan_ids": [b["scan_id"] for b in batch],
    }