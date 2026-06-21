"""Image-Video (I-V) cross-modal pairs from PVSG.

Construction (paper): the first frame of each video is the paired *image*; a
temporally non-adjacent segment is the paired *video*. 
The cross-modal association
is free — both views share PVSG's per-object ``object_id``, so image object i is
associated with video object j iff they have the same object_id.

A pairing "recipe" (which frames) is stored as JSON under data/multimodal/I-V/;
RGB frames + masks are read on the fly from the existing PVSG data.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .pvsg import SOURCE_DIRS, build_frame_targets


# --------------------------------------------------------------------------- #
# pairing recipe construction
# --------------------------------------------------------------------------- #
def build_iv_pairs(pvsg_ann, split: str, num_frames: int = 8,
                   image_frame: int = 0, gap_ratio: float = 0.5) -> List[Dict]:
    """Build I-V pairing recipes for a split (no disk reads; uses meta frame counts).

    image = frame ``image_frame`` (default 0); video = ``num_frames`` frames sampled
    from the temporally non-adjacent tail [gap_ratio*total, total-1].
    """
    if isinstance(pvsg_ann, str):
        with open(pvsg_ann) as f:
            pvsg_ann = json.load(f)
    vid2src = {}
    for src, parts in pvsg_ann["split"].items():
        for vid in parts.get(split, []):
            vid2src[vid] = SOURCE_DIRS[src]

    pairs = []
    for v in pvsg_ann["data"]:
        vid = v["video_id"]
        if vid not in vid2src:
            continue
        total = int(v["meta"]["num_frames"])
        start = max(image_frame + 1, int(total * gap_ratio))
        if start <= image_frame + 1 or start >= total:  # too short for a non-adjacent segment
            continue
        vf = torch.linspace(start, total - 1, num_frames).long().tolist()
        pairs.append({"video_id": vid, "source": vid2src[vid], "image_frame": image_frame,
                      "video_frames": vf, "num_frames_total": total})
    return pairs


def save_iv_pairs(pairs: List[Dict], out_path: str, split: str, num_frames: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"split": split, "num_frames": num_frames, "pairs": pairs}, f, indent=1)


# --------------------------------------------------------------------------- #
# cross-modal association (free, via shared object_id)
# --------------------------------------------------------------------------- #
def build_iv_association(image_object_ids: torch.Tensor,
                         video_object_ids: torch.Tensor) -> torch.Tensor:
    """GT association (M_img, M_vid): 1 where the two objects share an object_id."""
    if image_object_ids.numel() == 0 or video_object_ids.numel() == 0:
        return torch.zeros(image_object_ids.numel(), video_object_ids.numel())
    return (image_object_ids[:, None] == video_object_ids[None, :]).float()


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class IVPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_json: str, pvsg_ann, data_root: str,
                 preprocess=None, frame_reader=None, mask_size=(80, 80)):
        super().__init__()
        with open(pairs_json) as f:
            rec = json.load(f)
        self.pairs = rec["pairs"]
        if isinstance(pvsg_ann, str):
            with open(pvsg_ann) as f:
                pvsg_ann = json.load(f)
        self.thing = pvsg_ann["objects"]["thing"]
        self.stuff = pvsg_ann["objects"]["stuff"]
        self.object_classes = self.thing + self.stuff
        self.predicate_classes = pvsg_ann["relations"]
        self.pred_to_id = {p: i for i, p in enumerate(self.predicate_classes)}
        self.video = {v["video_id"]: v for v in pvsg_ann["data"]}
        self.data_root = data_root
        self.preprocess = preprocess
        self.frame_reader = frame_reader
        self.mask_size = tuple(mask_size)

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def _obj_to_label(self, video) -> Dict[int, int]:
        m = {}
        for o in video["objects"]:
            cat = o["category"]
            m[o["object_id"]] = (self.thing.index(cat) if o["is_thing"]
                                 else len(self.thing) + self.stuff.index(cat))
        return m

    def _relations(self, video):
        return [(int(s), int(o), self.pred_to_id[p], sp)
                for s, o, p, sp in video["relations"] if p in self.pred_to_id]

    def _frame_targets(self, mask_dir, fi, obj_to_label, relations):
        mpath = os.path.join(mask_dir, f"{fi:04d}.png")
        if os.path.isfile(mpath):
            mask = np.array(Image.open(mpath))
            return build_frame_targets(mask, obj_to_label, relations, fi, self.mask_size, return_ids=True)
        z = torch.zeros
        return z(0, dtype=torch.long), z(0, *self.mask_size), z(0, 3, dtype=torch.long), z(0, dtype=torch.long)

    def __getitem__(self, i: int) -> Dict:
        pair = self.pairs[i]
        vid, src = pair["video_id"], pair["source"]
        video = self.video[vid]
        obj_to_label = self._obj_to_label(video)
        relations = self._relations(video)
        mask_dir = os.path.join(self.data_root, src, "masks", vid)
        vpath = os.path.join(self.data_root, src, "videos", f"{vid}.mp4")

        # image (single frame)
        img_lab, img_msk, img_rel, img_ids = self._frame_targets(
            mask_dir, pair["image_frame"], obj_to_label, relations)
        image = None
        if self.frame_reader is not None:
            image = self.frame_reader(vpath, [pair["image_frame"]], self.preprocess)[0]  # (3,H,W)

        # video (segment): per-frame targets
        vid_lab, vid_msk, vid_rel, vid_ids = [], [], [], []
        for fi in pair["video_frames"]:
            l, m, r, ids = self._frame_targets(mask_dir, fi, obj_to_label, relations)
            vid_lab.append(l); vid_msk.append(m); vid_rel.append(r); vid_ids.append(ids)
        frames = self.frame_reader(vpath, pair["video_frames"], self.preprocess) \
            if self.frame_reader is not None else None

        return {
            "image": image, "image_labels": img_lab, "image_masks": img_msk,
            "image_relations": img_rel, "image_object_ids": img_ids,
            "frames": frames, "video_labels": vid_lab, "video_masks": vid_msk,
            "video_relations": vid_rel, "video_object_ids": vid_ids,
            "video_id": vid,
        }


def iv_collate(batch: List[Dict]) -> Dict:
    """Collate I-V pairs. image -> (B,3,H,W); video per-frame GT flattened to B*T."""
    images = torch.stack([b["image"] for b in batch]) if batch[0]["image"] is not None else None
    frames = torch.stack([b["frames"] for b in batch]) if batch[0]["frames"] is not None else None
    return {
        "images": images, "frames": frames,
        "image_labels": [b["image_labels"] for b in batch],
        "image_masks": [b["image_masks"] for b in batch],
        "image_relations": [b["image_relations"] for b in batch],
        "image_object_ids": [b["image_object_ids"] for b in batch],
        "video_labels": [x for b in batch for x in b["video_labels"]],          # B*T
        "video_masks": [x for b in batch for x in b["video_masks"]],
        "video_relations": [x for b in batch for x in b["video_relations"]],
        "video_object_ids": [x for b in batch for x in b["video_object_ids"]],
        "num_frames": len(batch[0]["video_labels"]),
        "video_ids": [b["video_id"] for b in batch],
    }