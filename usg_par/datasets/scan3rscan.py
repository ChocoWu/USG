"""Read 3RScan annotated point clouds for the 3DDSG (3DSSG) line.

Each scan's ``labels.instances.annotated.v2.ply`` stores per-vertex:
  x, y, z, red, green, blue, objectId (instance id), globalId (class id 1..527),
  NYU40, Eigen13, RIO27. We use it as the GT-segmented point cloud: xyz+rgb is the
  point-cloud input D, objectId gives GT instance masks, globalId the per-point class.
"""

import os
from dataclasses import dataclass

import numpy as np
from plyfile import PlyData


@dataclass
class ScanPointCloud:
    """One scan's point cloud (P points)."""
    xyz: np.ndarray        # (P, 3) float32
    rgb: np.ndarray        # (P, 3) float32 in [0,1]
    object_id: np.ndarray  # (P,) int32 instance id (0 = unannotated)
    global_id: np.ndarray  # (P,) int32 class id (1..527; 0 = unannotated)

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]


def read_3rscan_ply(ply_path: str) -> ScanPointCloud:
    """Parse a 3RScan annotated PLY (ascii or binary) into a ScanPointCloud."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    object_id = np.asarray(v["objectId"], dtype=np.int32)
    global_id = np.asarray(v["globalId"], dtype=np.int32)
    return ScanPointCloud(xyz=xyz, rgb=rgb, object_id=object_id, global_id=global_id)


def scan_ply_path(rscan_root: str, scan_id: str) -> str:
    return os.path.join(rscan_root, scan_id, "labels.instances.annotated.v2.ply")