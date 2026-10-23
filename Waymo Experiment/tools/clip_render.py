"""clip_render.py

Produce ONE continuous Waymo dawn/dusk clip as FOUR SEPARATE videos:
    1. lidar_bev      - top-down BEV of LiDAR points (#4C72B0 on near-black), no boxes
    2. rgb            - raw CAM_FRONT image, no overlays
    3. ssd_boxes_rgb  - CAM_FRONT + Fusion-SSD 3D pred boxes (projected via lidar2img)
    4. vlc_boxes_rgb  - CAM_FRONT + VLC-Fusion 3D pred boxes

Reuses tools/montage_infer.py (model loading + inference shims) and the
projection / box-corner helpers from tools/montage_render.py.

Fig 2 per-class colors (exact):
    Vehicle (0)    = yellow
    Pedestrian (1) = orange
    Cyclist (2)    = magenta

Class index -> name confirmed from configs/ensemble.py:
    class_names = ['Car', 'Pedestrian', 'Cyclist']  (0=Vehicle,1=Ped,2=Cyclist)

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/clip_render.py            # full run
    CUDA_VISIBLE_DEVICES=0 python tools/clip_render.py --no-cache # re-infer
"""
import os
import os.path as osp
import sys
import json
import pickle
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

PROJECT_ROOT = "/data/ataparia/VLCFusion/Waymo Experiment"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import tools.montage_infer as MI            # noqa: E402
import tools.montage_render as MR           # noqa: E402  (reuse geometry helpers)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OUT_DIR = "/data/ataparia/vlc_fusion_ECML/figures/waymo_montage/clip1"
PNG_DIRS = {
    "lidar_bev":     osp.join(OUT_DIR, "frames_lidar_bev"),
    "rgb":           osp.join(OUT_DIR, "frames_rgb"),
    "ssd_boxes_rgb": osp.join(OUT_DIR, "frames_ssd_boxes_rgb"),
    "vlc_boxes_rgb": osp.join(OUT_DIR, "frames_vlc_boxes_rgb"),
}
CACHE_PATH = osp.join(OUT_DIR, "_clip_infer_cache.pkl")
CLIP_INFO_PATH = osp.join(OUT_DIR, "_clip_info.json")

DATA_ROOT = MI.DATA_ROOT
IMG_DIR = osp.join(DATA_ROOT, "testing", "image_0")
LIDAR_DIR = osp.join(DATA_ROOT, "testing", "velodyne")
TEST_PKL = MI.TEST_PKL
CONDITIONS_JSONL = "conditions/dawn_dusk_testing.jsonl"
NIGHT_COND_IDX = 9  # nighttime flag in the 40-d conditions vector

# Chosen clip (see selection analysis in final report):
CONTEXT_NAME = "9114112687541091312_1100_000_1120_000"
WINDOW_START = 130       # frame offset within the timestamp-sorted context
WINDOW_LEN = 60          # cap
SCORE_THR = 0.3

# Fig 2 colors (RGB 0-1)
CLASS_COLORS = {
    0: (1.00, 0.85, 0.00),   # Vehicle    -> yellow
    1: (1.00, 0.55, 0.00),   # Pedestrian -> orange
    2: (1.00, 0.00, 1.00),   # Cyclist    -> magenta
}
CLASS_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
BEV_POINT_COLOR = "#4C72B0"   # seaborn/"facebook" blue
BEV_BG = "#070707"            # near-black
BEV_RANGE = 50.0              # +/- meters (x forward, y left)


# ----------------------------------------------------------------------------
# Clip selection
# ----------------------------------------------------------------------------
def _load_night_flags():
    flags = {}
    with open(CONDITIONS_JSONL) as f:
        for line in f:
            j = json.loads(line)
            conds = j["conditions"]
            flags[j["image_idx"]] = bool(conds[NIGHT_COND_IDX]) if len(conds) > NIGHT_COND_IDX else None
    return flags


def select_clip():
    with open(TEST_PKL, "rb") as f:
        info = pickle.load(f)
    ctx = defaultdict(list)
    for d in info["data_list"]:
        ctx[d["context_name"]].append(d)
    frames = sorted(ctx[CONTEXT_NAME], key=lambda d: d["timestamp"])
    clip = frames[WINDOW_START:WINDOW_START + WINDOW_LEN]
    sample_indices = [d["sample_idx"] for d in clip]
    ts = np.array([d["timestamp"] for d in clip], dtype=np.int64)
    gaps = np.diff(ts)
    med_gap = int(np.median(gaps)) if len(gaps) else 0
    night = _load_night_flags()
    n_night = sum(1 for si in sample_indices if night.get(si))
    return sample_indices, med_gap, n_night, info


# ----------------------------------------------------------------------------
# Renderers (one PNG per frame per stream)
# ----------------------------------------------------------------------------
def render_lidar_bev(sample_idx, out_path):
    pts = MR._load_points(sample_idx)           # (N,3)
    x, y = pts[:, 0], pts[:, 1]
    m = (np.abs(x) < BEV_RANGE) & (np.abs(y) < BEV_RANGE)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BEV_BG)
    fig.patch.set_facecolor(BEV_BG)
    # forward (x) -> up, left (y) -> left (so +y goes left): plot (-y, x)
    ax.scatter(-y[m], x[m], s=0.25, c=BEV_POINT_COLOR, alpha=0.85, linewidths=0)
    ax.plot(0, 0, marker="^", color="#d62728", markersize=9)  # ego
    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_path, dpi=110, facecolor=BEV_BG)
    plt.close(fig)


def render_rgb(sample_idx, out_path):
    img_path = osp.join(IMG_DIR, f"{sample_idx}.jpg")
    img = cv2.imread(img_path)
    cv2.imwrite(out_path, img)


def _draw_boxes(img_bgr, boxes, labels, lidar2img):
    """Reuse montage_render drawing but with Fig-2 colors. Returns BGR image."""
    img = img_bgr.copy()
    H, W = img.shape[:2]
    for box, lab in zip(boxes, labels):
        corners = MR._box_corners_3d(box)
        pts2d, depth = MR._project(corners, lidar2img)
        in_front = depth > 0.1
        if in_front.sum() < 2:
            continue
        col_rgb = CLASS_COLORS.get(int(lab), (1, 1, 1))
        col_bgr = tuple(int(255 * c) for c in col_rgb[::-1])
        for a, b in MR._BOX_EDGES:
            if not (in_front[a] and in_front[b]):
                continue
            pa, pb = pts2d[a], pts2d[b]
            if (abs(pa[0]) > 5 * W or abs(pa[1]) > 5 * H or
                    abs(pb[0]) > 5 * W or abs(pb[1]) > 5 * H):
                continue
            cv2.line(img, (int(round(pa[0])), int(round(pa[1]))),
                     (int(round(pb[0])), int(round(pb[1]))),
                     col_bgr, 2, cv2.LINE_AA)
    return img


def render_boxes_rgb(sample_idx, res, idx_by_sample, out_path):
    info = idx_by_sample[sample_idx]
    lidar2img = np.array(info["images"]["CAM_FRONT"]["lidar2img"])
    img_bgr = cv2.imread(osp.join(IMG_DIR, f"{sample_idx}.jpg"))
    out = _draw_boxes(img_bgr, res["bboxes_3d"], res["labels_3d"], lidar2img)
    cv2.imwrite(out_path, out)


# ----------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------
def _even(n):
    return n if n % 2 == 0 else n - 1


def encode_mp4(png_dir, n_frames, out_mp4, fps):
    pat = osp.join(png_dir, "%05d.png")
    # H.264 needs even dims; force via scale filter.
    cmd = [
        "/usr/bin/ffmpeg", "-y", "-framerate", str(fps), "-i", pat,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", out_mp4,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def encode_gif(png_dir, out_gif, fps, max_w=720):
    pat = osp.join(png_dir, "%05d.png")
    palette = out_gif + ".palette.png"
    vf = f"fps={fps},scale={max_w}:-1:flags=lanczos"
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-i", pat, "-vf", f"{vf},palettegen",
         palette], check=True, capture_output=True)
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-framerate", str(fps), "-i", pat,
         "-i", palette, "-lavfi", f"{vf} [x]; [x][1:v] paletteuse",
         out_gif], check=True, capture_output=True)
    if osp.exists(palette):
        os.remove(palette)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(use_cache=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    for d in PNG_DIRS.values():
        os.makedirs(d, exist_ok=True)

    sample_indices, med_gap, n_night, info = select_clip()
    idx_by_sample = {d["sample_idx"]: d for d in info["data_list"]}
    n = len(sample_indices)
    consecutive = abs(med_gap - 100000) < 5000
    fps = 10 if consecutive else 5
    print(f"Clip context: {CONTEXT_NAME}")
    print(f"Frames: {n}  median_gap_us: {med_gap}  night_frames: {n_night}/{n}  fps: {fps}")

    # --- Inference (cached) ---
    if use_cache and osp.exists(CACHE_PATH):
        print("Loading cached inference from", CACHE_PATH)
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        ssd_res, vlc_res = cache["ssd"], cache["vlc"]
    else:
        print("Running Fusion-SSD (Ensemble) inference...")
        ssd_res, _, _, _, _ = MI.run_inference("ensemble", sample_indices,
                                               score_thr=SCORE_THR)
        print("Running VLC-Fusion inference...")
        vlc_res, _, _, _, _ = MI.run_inference("vlc", sample_indices,
                                               score_thr=SCORE_THR)
        ssd_res = {k: {kk: v[kk] for kk in ("bboxes_3d", "scores_3d", "labels_3d")}
                   for k, v in ssd_res.items()}
        vlc_res = {k: {kk: v[kk] for kk in ("bboxes_3d", "scores_3d", "labels_3d")}
                   for k, v in vlc_res.items()}
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"ssd": ssd_res, "vlc": vlc_res,
                         "sample_indices": sample_indices}, f)
        print("Cached inference to", CACHE_PATH)

    # --- Render PNG sequences ---
    ssd_counts = defaultdict(int)
    vlc_counts = defaultdict(int)
    for i, sidx in enumerate(sample_indices):
        fn = f"{i:05d}.png"
        render_lidar_bev(sidx, osp.join(PNG_DIRS["lidar_bev"], fn))
        render_rgb(sidx, osp.join(PNG_DIRS["rgb"], fn))
        render_boxes_rgb(sidx, ssd_res[sidx], idx_by_sample,
                         osp.join(PNG_DIRS["ssd_boxes_rgb"], fn))
        render_boxes_rgb(sidx, vlc_res[sidx], idx_by_sample,
                         osp.join(PNG_DIRS["vlc_boxes_rgb"], fn))
        for c in (0, 1, 2):
            ssd_counts[c] += int((ssd_res[sidx]["labels_3d"] == c).sum())
            vlc_counts[c] += int((vlc_res[sidx]["labels_3d"] == c).sum())
        if i % 10 == 0 or i == n - 1:
            print(f"  rendered frame {i+1}/{n} (sidx={sidx})")

    print(f"SSD total preds: V={ssd_counts[0]} P={ssd_counts[1]} C={ssd_counts[2]}")
    print(f"VLC total preds: V={vlc_counts[0]} P={vlc_counts[1]} C={vlc_counts[2]}")

    # --- Encode ---
    out_mp4s, out_gifs = {}, {}
    for stream, png_dir in PNG_DIRS.items():
        mp4 = osp.join(OUT_DIR, f"{stream}.mp4")
        gif = osp.join(OUT_DIR, f"{stream}.gif")
        encode_mp4(png_dir, n, mp4, fps)
        encode_gif(png_dir, gif, fps)
        out_mp4s[stream], out_gifs[stream] = mp4, gif
        sz_mp4 = osp.getsize(mp4) / 1e6
        sz_gif = osp.getsize(gif) / 1e6
        print(f"  {stream}: mp4 {sz_mp4:.1f}MB | gif {sz_gif:.1f}MB")

    # --- Save clip metadata ---
    meta = {
        "context_name": CONTEXT_NAME,
        "split": "dawn_dusk testing",
        "n_frames": n,
        "median_gap_us": med_gap,
        "night_frames": n_night,
        "fps": fps,
        "score_thr": SCORE_THR,
        "sample_indices": sample_indices,
        "ssd_pred_counts": dict(ssd_counts),
        "vlc_pred_counts": dict(vlc_counts),
        "class_names": CLASS_NAMES,
        "class_colors_fig2": {CLASS_NAMES[c]: CLASS_COLORS[c] for c in (0, 1, 2)},
        "mp4s": out_mp4s,
        "gifs": out_gifs,
    }
    with open(CLIP_INFO_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print("\nDONE. Outputs in", OUT_DIR)
    print("mp4s:", out_mp4s)
    print("gifs:", out_gifs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    main(use_cache=not args.no_cache)
