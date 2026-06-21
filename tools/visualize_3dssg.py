"""Sanity-check a 3DSSG scan: render the point cloud colored by RGB / instance / class.

Headless (matplotlib Agg -> PNG). Also prints the per-instance class list and a few
relations so the geometry, masks, labels, and triplets can be eyeballed together.

  conda activate usg
  python tools/visualize_3dssg.py [scan_id]
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usg_par.datasets.scan3rscan import read_3rscan_ply, scan_ply_path  # noqa: E402

RSCAN = "data/3DSG/3RScan"
SSG = "data/3DSG/3DSSG"
OUT = "tools/vis_out"
MAX_POINTS = 20000          # subsample for render speed
ELEV, AZIM = 60, -60        # top-down-ish view


def _first_available_scan():
    rel = json.load(open(os.path.join(SSG, "relationships.json")))["scans"]
    for s in rel:
        if os.path.isfile(scan_ply_path(RSCAN, s["scan"])):
            return s["scan"]
    return None


def _scatter(ax, xyz, colors, title):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, depthshade=False, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    # equal aspect
    lim = np.array([xyz.min(0), xyz.max(0)])
    ctr = lim.mean(0); rng = (lim[1] - lim[0]).max() / 2
    for set_lim, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], ctr):
        set_lim(c - rng, c + rng)


def main():
    os.makedirs(OUT, exist_ok=True)
    sid = sys.argv[1] if len(sys.argv) > 1 else _first_available_scan()
    if sid is None:
        print("no downloaded scan available"); return 1
    print("scan:", sid)

    pc = read_3rscan_ply(scan_ply_path(RSCAN, sid))
    classes = [l.split("\t")[1] for l in open(os.path.join(SSG, "classes.txt")).read().splitlines() if l.strip()]
    obj_by_scan = {s["scan"]: s for s in json.load(open(os.path.join(SSG, "objects.json")))["scans"]}
    rel_by_scan = {s["scan"]: s for s in json.load(open(os.path.join(SSG, "relationships.json")))["scans"]}
    objs = {int(o["id"]): o["label"] for o in obj_by_scan[sid]["objects"]}

    # subsample
    rng = np.random.default_rng(0)
    idx = rng.choice(pc.num_points, min(MAX_POINTS, pc.num_points), replace=False)
    xyz, rgb, oid, gid = pc.xyz[idx], pc.rgb[idx], pc.object_id[idx], pc.global_id[idx]

    # color maps
    inst_ids = np.unique(oid)
    inst_color = {i: (rng.random(3) if i != 0 else np.array([0.6, 0.6, 0.6])) for i in inst_ids}
    inst_rgb = np.stack([inst_color[i] for i in oid])
    cls_ids = np.unique(gid)
    cls_color = {g: (rng.random(3) if g != 0 else np.array([0.6, 0.6, 0.6])) for g in cls_ids}
    cls_rgb = np.stack([cls_color[g] for g in gid])

    fig = plt.figure(figsize=(18, 6))
    for k, (col, title) in enumerate([(rgb, "RGB"),
                                      (inst_rgb, f"Instance ({len(inst_ids)-1} objects)"),
                                      (cls_rgb, f"Class ({len(cls_ids)} classes)")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        _scatter(ax, xyz, col, title)
    fig.suptitle(f"3DSSG scan {sid}", fontsize=13)
    out_png = os.path.join(OUT, f"{sid}.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print("saved:", out_png)

    # text summary
    print(f"\n{len(objs)} objects; per-instance point counts (top 12):")
    counts = [(int(i), int((pc.object_id == i).sum())) for i in inst_ids if i != 0]
    for i, c in sorted(counts, key=lambda x: -x[1])[:12]:
        print(f"  inst {i:3d}: {objs.get(i, '?'):20s} ({c} pts)")
    print("\nrelations (first 8):")
    for r in rel_by_scan[sid]["relationships"][:8]:
        s, o, pid, pname = r
        print(f"  ({objs.get(s,'?')}) -{pname}-> ({objs.get(o,'?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())