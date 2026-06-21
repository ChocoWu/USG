"""Project a 3RScan annotated point cloud into its 2D camera frames (for I-D).

A 3D object is "grounded" in a frame iff its points land on the visible surface (projected depth ~ measured depth), giving the cross-modal association by shared object_id. 
Reads color/depth/pose/intrinsics straight from each scan's sequence.zip.
"""

import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class SequenceInfo:
    color_w: int
    color_h: int
    color_K: np.ndarray      # (3,3)
    depth_w: int
    depth_h: int
    depth_K: np.ndarray
    depth_shift: float
    num_frames: int


def read_sequence_info(zf: zipfile.ZipFile) -> SequenceInfo:
    info = zf.read("_info.txt").decode()
    g = lambda k: re.search(rf"{k} = (.+)", info).group(1).strip()
    ck = np.array(g("m_calibrationColorIntrinsic").split(), float).reshape(4, 4)[:3, :3]
    dk = np.array(g("m_calibrationDepthIntrinsic").split(), float).reshape(4, 4)[:3, :3]
    return SequenceInfo(int(g("m_colorWidth")), int(g("m_colorHeight")), ck,
                        int(g("m_depthWidth")), int(g("m_depthHeight")), dk,
                        float(g("m_depthShift")), int(g("m_frames.size")))


def read_pose(zf, frame: str) -> np.ndarray:
    return np.array(zf.read(f"{frame}.pose.txt").decode().split(), float).reshape(4, 4)


def read_color(zf, frame: str) -> Image.Image:
    return Image.open(io.BytesIO(zf.read(f"{frame}.color.jpg"))).convert("RGB")


def read_depth(zf, frame: str, shift: float) -> np.ndarray:
    d = np.array(Image.open(io.BytesIO(zf.read(f"{frame}.depth.pgm")))).astype(np.float32)
    return d / shift     # meters


def _project(xyz: np.ndarray, w2c: np.ndarray, K, w: int, h: int):
    """world points -> (u, v, Z, in_frustum_mask) in a camera of size (w,h)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    with np.errstate(all="ignore"):
        pc = (w2c @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
        z = pc[:, 2]
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, inside


def visible_object_masks(pc, zf, frame: str, info: SequenceInfo,
                         mask_size=(80, 80), min_points: int = 30, depth_thr: float = 0.15
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Visible (grounded) object_ids and their 2D masks for one frame.

    Visibility = on the depth surface (|projected Z - measured depth| < depth_thr).
    Masks are rasterized in the color frame and resized to ``mask_size``.

    Returns (object_ids (M,) long, masks (M, *mask_size) float).
    """
    pose = read_pose(zf, frame)
    w2c = np.linalg.inv(pose)
    depth = read_depth(zf, frame, info.depth_shift)

    # occlusion via depth frame
    ud, vd, zd, ind = _project(pc.xyz, w2c, info.depth_K, info.depth_w, info.depth_h)
    surf = np.zeros(len(pc.xyz), bool)
    di, vi, ui = np.where(ind)[0], vd[ind].astype(int), ud[ind].astype(int)
    meas = depth[vi, ui]
    on_surface = (meas > 0.1) & (np.abs(zd[ind] - meas) < depth_thr)
    surf[di[on_surface]] = True

    # rasterize surface points into the color frame, per object
    uc, vc, zc, inc = _project(pc.xyz, w2c, info.color_K, info.color_w, info.color_h)
    keep = surf & inc
    ids_all = pc.object_id[keep]
    uc_k, vc_k = uc[keep].astype(int), vc[keep].astype(int)

    object_ids, masks = [], []
    for oid in np.unique(ids_all):
        if oid == 0:
            continue
        sel = ids_all == oid
        if sel.sum() < min_points:
            continue
        m = torch.zeros(info.color_h, info.color_w)
        m[vc_k[sel], uc_k[sel]] = 1.0
        # dilate-free resize (nearest) to mask resolution
        m = F.interpolate(m[None, None], size=mask_size, mode="bilinear", align_corners=False)[0, 0]
        object_ids.append(int(oid))
        masks.append((m > 0).float())
    if object_ids:
        return torch.tensor(object_ids, dtype=torch.long), torch.stack(masks)
    return torch.zeros(0, dtype=torch.long), torch.zeros(0, *mask_size)


def open_sequence(scan_dir: str) -> Tuple[zipfile.ZipFile, SequenceInfo]:
    zf = zipfile.ZipFile(f"{scan_dir}/sequence.zip")
    return zf, read_sequence_info(zf)