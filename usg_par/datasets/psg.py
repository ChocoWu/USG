"""PSG (Panoptic Scene Graph) dataset loader for the image SGDet task (Table 2).

Annotation format (data/psg/psg.json):
  data: per-image list; each item has file_name, pan_seg_file_name, height, width,
        image_id, segments_info (list of {id, category_id, isthing, ...}), and
        relations (list of [subject_seg_idx, object_seg_idx, predicate_id]).
  thing_classes (80) + stuff_classes (53) = 133 object classes; category_id is
  ALREADY the combined 0..132 index. predicate_classes: 56.
  Split: image_id in test_image_ids -> test, else train.

Image data lives in coco.zip (extract to data/psg/coco -> train2017/, panoptic_train2017/).
If the image root is absent the dataset runs in annotation-only mode (image/masks None),
which is enough to test target construction.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def decode_panoptic_masks(pan_seg: np.ndarray, segments_info: List[dict]):
    """Decode a COCO-panoptic PNG into per-segment binary masks + labels.

    Args:
        pan_seg: (H, W, 3+) uint8 array; segment id = R + G*256 + B*256^2.
        segments_info: list of {id, category_id, ...} in annotation order.

    Returns:
        masks: (M, H, W) bool, labels: (M,) long (combined 0..132 category_id).
    """
    rgb = pan_seg[..., :3].astype(np.int64)
    seg_ids = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 256 * 256
    masks, labels = [], []
    for seg in segments_info:
        masks.append(torch.from_numpy(seg_ids == seg["id"]))
        labels.append(seg["category_id"])
    if masks:
        return torch.stack(masks), torch.tensor(labels, dtype=torch.long)
    h, w = seg_ids.shape
    return torch.zeros(0, h, w, dtype=torch.bool), torch.zeros(0, dtype=torch.long)


def _resize_masks(masks: torch.Tensor, size) -> torch.Tensor:
    """Nearest-resize (M,H,W) bool masks to (M, *size) float in {0,1}."""
    if masks.numel() == 0:
        return torch.zeros(0, *size)
    m = F.interpolate(masks.float().unsqueeze(1), size=size, mode="nearest").squeeze(1)
    return m


class PSGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann,                                   # path to psg.json OR an already-loaded dict
        image_root: Optional[str] = None,      # e.g. data/psg/coco
        preprocess=None,                       # OpenCLIP image transform (PIL -> tensor)
        split: str = "train",
        mask_size=(80, 80),                    # GT mask resolution (match pred mask_features)
        load_masks: bool = True,
    ):
        super().__init__()
        if isinstance(ann, str):
            with open(ann) as f:
                ann = json.load(f)
        self.thing_classes = ann["thing_classes"]
        self.stuff_classes = ann["stuff_classes"]
        self.object_classes = self.thing_classes + self.stuff_classes      # 133
        self.predicate_classes = ann["predicate_classes"]                   # 56
        test_ids = set(ann["test_image_ids"])
        keep_test = split == "test"
        self.items = [
            it for it in ann["data"]
            if (it["image_id"] in test_ids) == keep_test
        ]
        self.image_root = image_root
        self.preprocess = preprocess
        self.split = split
        self.mask_size = tuple(mask_size)
        self.load_masks = load_masks

    @property
    def num_object_classes(self) -> int:
        return len(self.object_classes)

    @property
    def num_predicates(self) -> int:
        return len(self.predicate_classes)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        it = self.items[i]
        seg_info = it["segments_info"]
        labels = torch.tensor([s["category_id"] for s in seg_info], dtype=torch.long)
        relations = torch.tensor(it["relations"], dtype=torch.long) if it["relations"] \
            else torch.zeros(0, 3, dtype=torch.long)

        image, masks = None, None
        has_images = self.image_root is not None and os.path.isdir(self.image_root)
        if has_images and self.preprocess is not None:
            img = Image.open(os.path.join(self.image_root, it["file_name"])).convert("RGB")
            image = self.preprocess(img)
        if has_images and self.load_masks:
            pan = np.array(Image.open(os.path.join(self.image_root, it["pan_seg_file_name"])))
            m, mlabels = decode_panoptic_masks(pan, seg_info)
            masks = _resize_masks(m, self.mask_size)
            labels = mlabels  # consistent ordering with masks

        return {
            "image": image,
            "labels": labels,
            "masks": masks,
            "relations": relations,
            "image_id": it["image_id"],
        }


def psg_collate(batch: List[Dict]) -> Dict:
    """Collate variable-#object items. Images stack (same size); rest stay as lists."""
    images = [b["image"] for b in batch]
    images = torch.stack(images) if images[0] is not None else None
    return {
        "images": images,
        "labels": [b["labels"] for b in batch],
        "masks": [b["masks"] for b in batch] if batch[0]["masks"] is not None else None,
        "relations": [b["relations"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }