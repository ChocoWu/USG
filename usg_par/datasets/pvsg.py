"""PVSG (Panoptic Video Scene Graph) dataset.

Annotation (data/pvsg/pvsg.json):
  objects: {thing: [...115], stuff: [...11]} -> 126 combined object classes.
  relations: [...57] predicate names.
  split: {vidor:{train,val}, epic_kitchen:{...}, ego4d:{...}} of video_ids.
  data: per-video {video_id, meta{height,width,fps,num_frames}, objects:[{object_id,
        category, is_thing}], relations:[[sub_id, obj_id, predicate_name, [[s,e],...]]]}.


A relation holds on frame f iff some span [s,e] covers f. 
We sample T frames per
video and turn each into a per-frame scene graph (labels + masks + intra-frame
relations).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SOURCE_DIRS = {"vidor": "VidOR", "epic_kitchen": "EpicKitchen", "ego4d": "Ego4D"}


def _resize_masks(masks: torch.Tensor, size) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros(0, *size)
    return F.interpolate(masks.float().unsqueeze(1), size=size, mode="nearest").squeeze(1)


def build_frame_targets(mask: np.ndarray, obj_to_label: Dict[int, int],
                        relations: List, frame_idx: int, mask_size=(80, 80),
                        return_ids: bool = False):
    """Turn one frame's object-id map + temporal relations into per-frame targets.

    Args:
        mask: (H, W) object-id map (pixel == object_id).
        obj_to_label: object_id -> combined class label.
        relations: list of (sub_id, obj_id, predicate_id, spans) for the video.
        frame_idx: index of this frame.
        mask_size: GT mask resolution.
        return_ids: if True, also return the present object_ids (for cross-modal assoc).

    Returns:
        labels (M,), masks (M, *mask_size) float, frame_relations (R, 3) over local idx
        [, object_ids (M,) long  if return_ids].
    """
    present = [oid for oid in np.unique(mask).tolist() if oid != 0 and oid in obj_to_label]
    id_to_local = {oid: i for i, oid in enumerate(present)}
    labels = torch.tensor([obj_to_label[oid] for oid in present], dtype=torch.long)
    if present:
        m = torch.stack([torch.from_numpy(mask == oid) for oid in present])
        masks = _resize_masks(m, mask_size)
    else:
        masks = torch.zeros(0, *mask_size)

    rels = []
    for sub_id, obj_id, pred_id, spans in relations:
        if sub_id in id_to_local and obj_id in id_to_local \
                and any(s <= frame_idx <= e for s, e in spans):
            rels.append([id_to_local[sub_id], id_to_local[obj_id], pred_id])
    frame_relations = torch.tensor(rels, dtype=torch.long) if rels else torch.zeros(0, 3, dtype=torch.long)
    if return_ids:
        object_ids = torch.tensor(present, dtype=torch.long)
        return labels, masks, frame_relations, object_ids
    return labels, masks, frame_relations


class PVSGDataset(torch.utils.data.Dataset):
    def __init__(self, ann, data_root: str, split: str = "train", num_frames: int = 8,
                 mask_size=(80, 80), preprocess=None, frame_reader=None, load_frames: bool = True):
        super().__init__()
        if isinstance(ann, str):
            with open(ann) as f:
                ann = json.load(f)
        self.thing = ann["objects"]["thing"]
        self.stuff = ann["objects"]["stuff"]
        self.object_classes = self.thing + self.stuff                 # 126
        self.predicate_classes = ann["relations"]                     # 57
        self.pred_to_id = {p: i for i, p in enumerate(self.predicate_classes)}

        # video_id -> source dir (from split)
        vid2src = {}
        for src, parts in ann["split"].items():
            wanted = parts.get(split, [])
            for vid in wanted:
                vid2src[vid] = SOURCE_DIRS[src]
        self.items = [v for v in ann["data"] if v["video_id"] in vid2src]
        self.vid2src = vid2src

        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.mask_size = tuple(mask_size)
        self.preprocess = preprocess
        self.frame_reader = frame_reader
        self.load_frames = load_frames

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.items)

    def _obj_to_label(self, video) -> Dict[int, int]:
        m = {}
        for o in video["objects"]:
            cat = o["category"]
            m[o["object_id"]] = (self.thing.index(cat) if o["is_thing"]
                                 else len(self.thing) + self.stuff.index(cat))
        return m

    def _relations(self, video) -> List:
        out = []
        for sub_id, obj_id, pred_name, spans in video["relations"]:
            if pred_name in self.pred_to_id:
                out.append((sub_id, obj_id, self.pred_to_id[pred_name], spans))
        return out

    def _sample_frame_indices(self, total: int) -> List[int]:
        if total <= self.num_frames:
            return list(range(total))
        return torch.linspace(0, total - 1, self.num_frames).long().tolist()

    def __getitem__(self, i: int) -> Dict:
        video = self.items[i]
        vid = video["video_id"]
        src_dir = self.vid2src[vid]
        obj_to_label = self._obj_to_label(video)
        relations = self._relations(video)
        mask_dir = os.path.join(self.data_root, src_dir, "masks", vid)

        total = video["meta"]["num_frames"]
        if os.path.isdir(mask_dir):
            total = len([f for f in os.listdir(mask_dir) if f.endswith(".png")]) or total
        frame_idxs = self._sample_frame_indices(total)

        labels_t, masks_t, rels_t = [], [], []
        for fi in frame_idxs:
            mpath = os.path.join(mask_dir, f"{fi:04d}.png")
            if os.path.isfile(mpath):
                mask = np.array(Image.open(mpath))
                lab, msk, rel = build_frame_targets(mask, obj_to_label, relations, fi, self.mask_size)
            else:
                lab = torch.zeros(0, dtype=torch.long)
                msk = torch.zeros(0, *self.mask_size)
                rel = torch.zeros(0, 3, dtype=torch.long)
            labels_t.append(lab); masks_t.append(msk); rels_t.append(rel)

        frames = None
        if self.load_frames and self.frame_reader is not None:
            vpath = os.path.join(self.data_root, src_dir, "videos", f"{vid}.mp4")
            frames = self.frame_reader(vpath, frame_idxs, self.preprocess)  # (T,3,H,W)

        return {
            "frames": frames, "labels": labels_t, "masks": masks_t,
            "relations": rels_t, "video_id": vid, "frame_idxs": frame_idxs,
        }


def pvsg_collate(batch: List[Dict]) -> Dict:
    """Collate videos. frames -> (B,T,3,H,W); per-frame GT flattened to B*T lists."""
    frames = [b["frames"] for b in batch]
    frames = torch.stack(frames) if frames[0] is not None else None
    # flatten per-frame GT across the batch (row-major: video b, frame t)
    labels, masks, relations = [], [], []
    for b in batch:
        labels.extend(b["labels"])
        masks.extend(b["masks"])
        relations.extend(b["relations"])
    return {
        "frames": frames, "labels": labels,
        "masks": masks, "relations": relations,
        "video_ids": [b["video_id"] for b in batch],
        "num_frames": len(batch[0]["labels"]),
    }