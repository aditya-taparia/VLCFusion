"""montage_render.py

Build a 3-column qualitative montage (and assembled GIF) comparing
Fusion-SSD (Ensemble) vs VLC-Fusion 3D detection on Waymo dawn/dusk frames.

Layout per frame (rows = camera / BEV, columns = GT / Fusion-SSD / VLC-Fusion):
    Col 1  Ground Truth      : camera + 3D GT boxes,    BEV + GT boxes
    Col 2  Fusion SSD        : camera + SSD pred boxes,  BEV + SSD boxes
    Col 3  VLC Fusion (Ours) : camera + VLC pred boxes,  BEV + VLC boxes

Classes are colored consistently: Vehicle/Car, Pedestrian, Cyclist.
3D boxes are projected to the image via the per-frame lidar2img matrix.
BEV is a top-down x-y scatter of the LiDAR points with rotated box rectangles.

Outputs (under /data/ataparia/vlc_fusion_ECML/figures/waymo_montage/):
    montage_<idx>.png   (one per frame)
    waymo_ssd_vs_vlc.gif (cycles through the frames)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/montage_render.py
"""
import os
import os.path as osp
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
import cv2

PROJECT_ROOT = "/data/ataparia/VLCFusion/Waymo Experiment"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import tools.montage_infer as MI  # noqa: E402

OUT_DIR = "/data/ataparia/vlc_fusion_ECML/figures/waymo_montage"
DATA_ROOT = MI.DATA_ROOT
IMG_DIR = osp.join(DATA_ROOT, "testing", "image_0")
LIDAR_DIR = osp.join(DATA_ROOT, "testing", "velodyne")

# Frames chosen: distinct dark dawn/dusk scenes flagged nighttime by GPT-4o,
# with a balanced mix of vehicles/pedestrians and (where available) a cyclist.
FRAMES = [20004652, 20004984, 2000830, 20002096, 2000629]
SCORE_THR = 0.3

# class colors (RGB 0-1) consistent across all panels
CLASS_COLORS = {
    0: (0.13, 0.70, 1.00),   # Car/Vehicle  - cyan/blue
    1: (1.00, 0.45, 0.10),   # Pedestrian   - orange
    2: (0.20, 0.90, 0.30),   # Cyclist      - green
}
CLASS_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}

# BEV view range (meters) in LiDAR frame (x forward, y left)
BEV_X = (-50, 50)
BEV_Y = (-50, 50)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _box_corners_3d(box):
    """Return 8 corners (8,3) in LiDAR frame for a box [x,y,z,l,w,h,yaw].

    z is the box bottom center (mmdet3d LiDARInstance3DBoxes convention).
    Corner order matches draw order used below.
    """
    x, y, z, l, w, h, yaw = box[:7]
    # corners in box frame (origin = bottom center)
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_c = np.array([   0,    0,    0,    0,    h,    h,    h,    h])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners.T  # (8,3)


def _project(corners_3d, lidar2img):
    """Project (N,3) LiDAR points to image (N,2) plus depth (N,)."""
    n = corners_3d.shape[0]
    pts4 = np.concatenate([corners_3d, np.ones((n, 1))], axis=1)  # (N,4)
    l2i = np.array(lidar2img)
    if l2i.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]])
        l2i = np.concatenate([l2i, bottom], axis=0)
    uvw = pts4 @ l2i.T  # (N,4)
    depth = uvw[:, 2]
    uvw[:, 2] = np.clip(uvw[:, 2], 1e-5, 1e6)
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    return np.stack([u, v], axis=1), depth


# 12 edges of a cuboid given the corner order in _box_corners_3d
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def draw_boxes_on_image(img_bgr, boxes, labels, lidar2img, thickness=2):
    """Draw projected 3D boxes on a copy of the BGR image. Returns RGB image."""
    img = img_bgr.copy()
    H, W = img.shape[:2]
    for box, lab in zip(boxes, labels):
        corners = _box_corners_3d(box)
        # skip boxes fully behind the camera
        _, depth = _project(corners, lidar2img)
        if np.all(depth <= 0.1):
            continue
        pts2d, _ = _project(corners, lidar2img)
        col_rgb = CLASS_COLORS.get(int(lab), (1, 1, 1))
        col_bgr = tuple(int(255 * c) for c in col_rgb[::-1])
        # only draw if at least one corner is reasonably in front and near frame
        in_front = depth > 0.1
        if in_front.sum() < 2:
            continue
        for a, b in _BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            pa = pts2d[a]
            pb = pts2d[b]
            # clip absurd coordinates
            if (abs(pa[0]) > 5 * W or abs(pa[1]) > 5 * H or
                    abs(pb[0]) > 5 * W or abs(pb[1]) > 5 * H):
                continue
            cv2.line(img, (int(round(pa[0])), int(round(pa[1]))),
                     (int(round(pb[0])), int(round(pb[1]))),
                     col_bgr, thickness, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_points(sample_idx):
    info_path = osp.join(LIDAR_DIR, f"{sample_idx}.bin")
    pts = np.fromfile(info_path, dtype=np.float32).reshape(-1, 6)
    return pts[:, :3]


def draw_bev(ax, points, boxes, labels, title=None):
    """Render a top-down BEV: LiDAR scatter + rotated box rectangles."""
    x, y = points[:, 0], points[:, 1]
    m = (x > BEV_X[0]) & (x < BEV_X[1]) & (y > BEV_Y[0]) & (y < BEV_Y[1])
    # In BEV image: forward (x) -> up, left (y) -> left. Plot y on horizontal
    # (flipped so +y is left), x on vertical.
    ax.scatter(-y[m], x[m], s=0.12, c="#bfbfbf", alpha=0.5, linewidths=0)
    for box, lab in zip(boxes, labels):
        corners = _box_corners_3d(box)[:4, :2]  # bottom 4 corners x,y
        poly_xy = np.stack([-corners[:, 1], corners[:, 0]], axis=1)
        col = CLASS_COLORS.get(int(lab), (1, 1, 1))
        ax.add_patch(MplPolygon(poly_xy, closed=True, fill=False,
                                edgecolor=col, linewidth=1.4))
        # heading tick: midpoint of front edge (corners 0-1)
        front_mid = (corners[0, :2] + corners[1, :2]) / 2.0
        cen = corners[:, :2].mean(axis=0)
        ax.plot([-cen[1], -front_mid[1]], [cen[0], front_mid[0]],
                c=col, linewidth=1.0)
    ax.set_xlim(-BEV_Y[1], -BEV_Y[0])
    ax.set_ylim(BEV_X[0], BEV_X[1])
    ax.set_aspect("equal")
    ax.set_facecolor("#0e0e0e")
    ax.set_xticks([])
    ax.set_yticks([])
    # ego marker at origin
    ax.plot(0, 0, marker="^", color="red", markersize=7)
    if title:
        ax.set_title(title, fontsize=12, color="black", pad=6)


def get_gt(sample_idx, idx_by_sample):
    """Return (boxes (N,7), labels (N,)) GT in LiDAR frame, classes 0/1/2 only."""
    info = idx_by_sample[sample_idx]
    boxes, labels = [], []
    for inst in info["instances"]:
        lab = inst["bbox_label_3d"]
        if lab < 0 or lab > 2:
            continue
        boxes.append(np.array(inst["bbox_3d"], dtype=np.float32))
        labels.append(int(lab))
    if boxes:
        return np.stack(boxes), np.array(labels)
    return np.zeros((0, 7), np.float32), np.zeros((0,), int)


def _legend_handles():
    return [Line2D([0], [0], color=CLASS_COLORS[c], lw=3, label=CLASS_NAMES[c])
            for c in (0, 1, 2)]


def render_frame(sample_idx, gt, ssd, vlc, idx_by_sample):
    """Build a 2x3 montage figure for one frame and return the PNG path."""
    info = idx_by_sample[sample_idx]
    lidar2img = np.array(info["images"]["CAM_FRONT"]["lidar2img"])
    img_path = osp.join(IMG_DIR, f"{sample_idx}.jpg")
    img_bgr = cv2.imread(img_path)
    points = _load_points(sample_idx)

    cols = [
        ("Ground Truth", gt["boxes"], gt["labels"]),
        ("Fusion SSD", ssd["bboxes_3d"], ssd["labels_3d"]),
        ("VLC Fusion (Ours)", vlc["bboxes_3d"], vlc["labels_3d"]),
    ]

    fig = plt.figure(figsize=(19, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.08,
                          wspace=0.04, left=0.01, right=0.99,
                          top=0.88, bottom=0.06)

    for ci, (title, boxes, labels) in enumerate(cols):
        # top: camera with projected boxes
        ax_im = fig.add_subplot(gs[0, ci])
        img_rgb = draw_boxes_on_image(img_bgr, boxes, labels, lidar2img)
        ax_im.imshow(img_rgb)
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        n_by = {c: int((np.array(labels) == c).sum()) for c in (0, 1, 2)} \
            if len(labels) else {0: 0, 1: 0, 2: 0}
        sub = f"V:{n_by[0]}  P:{n_by[1]}  C:{n_by[2]}"
        tcolor = (0.0, 0.35, 0.0) if ci == 2 else "black"
        ax_im.set_title(f"{title}\n{sub}", fontsize=14,
                        fontweight="bold" if ci == 2 else "normal", pad=8,
                        color=tcolor)
        # bottom: BEV
        ax_bev = fig.add_subplot(gs[1, ci])
        draw_bev(ax_bev, points, boxes, labels)

    # figure legend
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"Waymo dawn/dusk (low-light)  -  frame {sample_idx}   "
        f"[score >= {SCORE_THR}]",
        fontsize=16, y=0.975, fontweight="bold")

    out_path = osp.join(OUT_DIR, f"montage_{sample_idx}.png")
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


CACHE_PATH = osp.join(OUT_DIR, "_infer_cache.pkl")


def _serializable(res):
    """Strip non-picklable LiDARInstance3DBoxes objects from results."""
    out = {}
    for sidx, r in res.items():
        out[sidx] = {
            "bboxes_3d": r["bboxes_3d"],
            "scores_3d": r["scores_3d"],
            "labels_3d": r["labels_3d"],
        }
    return out


def main(use_cache=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    idx_by_sample = MI._load_infos()

    if use_cache and osp.exists(CACHE_PATH):
        print("Loading cached inference results from", CACHE_PATH)
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        ssd_res, vlc_res = cache["ssd"], cache["vlc"]
    else:
        # Run both models once over all frames (build each model a single time).
        print("Running Fusion-SSD (Ensemble) inference...")
        ssd_res, _, _, _, _ = MI.run_inference("ensemble", FRAMES,
                                               score_thr=SCORE_THR)
        print("Running VLC-Fusion inference...")
        vlc_res, _, _, _, _ = MI.run_inference("vlc", FRAMES,
                                               score_thr=SCORE_THR)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"ssd": _serializable(ssd_res),
                         "vlc": _serializable(vlc_res)}, f)
        print("Cached inference results to", CACHE_PATH)

    png_paths = []
    for sidx in FRAMES:
        gt_boxes, gt_labels = get_gt(sidx, idx_by_sample)
        gt = {"boxes": gt_boxes, "labels": gt_labels}
        p = render_frame(sidx, gt, ssd_res[sidx], vlc_res[sidx], idx_by_sample)
        print("wrote", p)
        png_paths.append(p)

    # Assemble GIF
    gif_path = osp.join(OUT_DIR, "waymo_ssd_vs_vlc.gif")
    try:
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in png_paths]
        # unify size to the first frame
        base = imgs[0].size
        imgs = [im.resize(base) if im.size != base else im for im in imgs]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=1500, loop=0)
        print("wrote", gif_path)
    except Exception as e:
        print("GIF assembly failed:", e)
        gif_path = None

    print("\nDONE")
    print("PNGs:")
    for p in png_paths:
        print("  ", p)
    print("GIF:", gif_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="force re-running inference instead of using cache")
    args = ap.parse_args()
    main(use_cache=not args.no_cache)
