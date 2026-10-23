"""clip_batch.py

Batch-generate qualitative Waymo clips for 3 scenarios (dawn_dusk, day, night).

Reuses the existing working pipeline:
  - tools/montage_infer.py  : all mmdet3d shims / registrations (imported for side effects)
  - tools/montage_render.py : box-corner + projection geometry helpers (MR.*)

Key differences from clip_render.py (which did ONE hardcoded dawn_dusk clip):
  - Generalised to TWO source pkls / splits:
      * dawn_dusk : /data/.../dawn_dusk-frames/waymo_infos_test.pkl  (testing/ images, prefix '2')
      * day/night : /data/.../night-day-frames/waymo_infos_val.pkl   (training/ images, prefix '1')
  - VLC model is built ONCE with all three condition jsonls so routing by the
    first char of the image basename works for both '1' (val dict) and '2' (test dict).
  - Ensemble (Fusion-SSD) model is rebuilt per split (different data_root / image dir).
  - Per clip writes 4 mp4 + data.pkl (rich, unfiltered preds) + info.json.
  - Incremental, resumable manifest.json. Skips already-completed clips.

GPU 0 only (set CUDA_VISIBLE_DEVICES=0 in the environment before launching).

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/clip_batch.py --validate   # clip_01 of each scenario
  CUDA_VISIBLE_DEVICES=0 python tools/clip_batch.py              # full 30-clip run
  CUDA_VISIBLE_DEVICES=0 python tools/clip_batch.py --scenario day  # one scenario
"""
import os
import os.path as osp
import sys
import json
import pickle
import copy
import argparse
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

# montage_infer performs ALL the mmdet3d / mmdet registrations & shims we need.
import tools.montage_infer as MI       # noqa: E402
import tools.montage_render as MR      # noqa: E402  (geometry helpers)
import tools.box_draw as BD            # noqa: E402  (front-face-shaded 3D boxes)

from mmengine.config import Config            # noqa: E402
from mmengine.dataset import pseudo_collate    # noqa: E402
from mmdet3d.apis import init_model            # noqa: E402
from mmdet3d.registry import DATASETS          # noqa: E402
import torch                                   # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_ROOT = "/data/ataparia/vlc_fusion_ECML/figures/waymo_montage/clips"
MANIFEST_PATH = osp.join(OUT_ROOT, "manifest.json")

FFMPEG = "/usr/bin/ffmpeg"
FPS = 10
VIDEO_SCORE_THR = 0.3      # box render threshold
SAVE_SCORE_THR = 0.0       # keep ALL preds in data.pkl (no filter)
WINDOW_LEN = 60            # max contiguous frames per clip
N_CLIPS = 10               # per scenario
NIGHT_IDX0 = 8             # array index for 1-based condition #9 (nighttime)

# 10-d VLC condition sampling (mirrors VLCFusion.load_conditions, n_conditions=10)
COND_INDICES = np.array([31, 16, 9, 40, 27, 6, 3, 23, 10, 1])

# Fig-2 exact colors (RGB 0-1)
CLASS_COLORS = {
    0: (1.00, 0.85, 0.00),   # Vehicle    -> yellow
    1: (1.00, 0.55, 0.00),   # Pedestrian -> orange
    2: (1.00, 0.00, 1.00),   # Cyclist    -> magenta
}
CLASS_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
BEV_POINT_COLOR = "#4C72B0"
BEV_BG = "#070707"
BEV_RANGE = 50.0
BEV_IMG = 880   # 8 in * 110 dpi, axes fill the figure (add_axes([0,0,1,1]))

# Constant transform to overlay 3D boxes onto the SAVED lidar_bev frames without
# raw points. lidar (x fwd, y left) -> BEV pixel (u col, v row):
#   u = (BEV_RANGE - y)/(2*BEV_RANGE)*BEV_IMG ;  v = (BEV_RANGE - x)/(2*BEV_RANGE)*BEV_IMG
BEV_PARAMS = dict(range_m=BEV_RANGE, img_w=BEV_IMG, img_h=BEV_IMG,
                  x_axis="up", y_axis="left", ego="center",
                  u_expr="(range_m - y)/(2*range_m)*img_w",
                  v_expr="(range_m - x)/(2*range_m)*img_h")

# Per-scenario data sources.
SCENARIOS = {
    "dawn_dusk": dict(
        pkl="/data/ataparia/LidarTraining/dawn_dusk-frames/waymo_infos_test.pkl",
        data_root="/data/ataparia/LidarTraining/dawn_dusk-frames/",
        img_subdir="testing/image_0",
        lidar_subdir="testing/velodyne",
        ann_file="waymo_infos_test.pkl",
        pts_prefix="testing/velodyne",
        img_prefix="testing/image_0",
        cond_jsonl="conditions/dawn_dusk_testing.jsonl",
        cond_dict="test",          # routes via test_conditions_dict (prefix '2')
        prefer_continuous=True,    # true 10 fps source
    ),
    "day": dict(
        pkl="/data/ataparia/LidarTraining/night-day-frames/waymo_infos_val.pkl",
        data_root="/data/ataparia/LidarTraining/night-day-frames/",
        img_subdir="training/image_0",
        lidar_subdir="training/velodyne",
        ann_file="waymo_infos_val.pkl",
        pts_prefix="training/velodyne",
        img_prefix="training/image_0",
        cond_jsonl="conditions/night-day_validation.jsonl",
        cond_dict="val",           # routes via val_conditions_dict (prefix '1')
        prefer_continuous=False,   # subsampled source (~700 ms gaps)
    ),
    "night": dict(
        pkl="/data/ataparia/LidarTraining/night-day-frames/waymo_infos_val.pkl",
        data_root="/data/ataparia/LidarTraining/night-day-frames/",
        img_subdir="training/image_0",
        lidar_subdir="training/velodyne",
        ann_file="waymo_infos_val.pkl",
        pts_prefix="training/velodyne",
        img_prefix="training/image_0",
        cond_jsonl="conditions/night-day_validation.jsonl",
        cond_dict="val",
        prefer_continuous=False,
    ),
}


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
def _load_cond_jsonl(path):
    """basename -> 40-d bool array."""
    d = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            d[osp.basename(j["image_path"])] = np.array(j["conditions"])
    return d


def cond10(arr40):
    """Apply the model's 10-d sampling to a 40-d condition array."""
    return arr40[COND_INDICES - 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
def build_vlc_model():
    """Build VLC ONCE with val+test+train condition dicts so routing by basename
    prefix works for both day/night (prefix '1' -> val dict) and dawn_dusk
    (prefix '2' -> test dict)."""
    cfg = Config.fromfile(osp.join(PROJECT_ROOT, MI.VLC_CFG))
    cfg.model.lidar_model_path = "/data/ataparia/mmdet_lidar/work_dirs/second/epoch_100.pth"
    cfg.model.train_jsonl_file = "conditions/night-day_training.jsonl"
    cfg.model.val_jsonl_file = "conditions/night-day_validation.jsonl"
    cfg.model.test_jsonl_file = "conditions/dawn_dusk_testing.jsonl"
    model = init_model(cfg, MI.VLC_CKPT, device="cuda:0")
    model.eval()
    return model


def build_ensemble_model(sc):
    """Build Fusion-SSD (Ensemble) for a given scenario's split."""
    cfg = Config.fromfile(osp.join(PROJECT_ROOT, MI.ENSEMBLE_CFG))
    cfg.model.lidar_model_path = "/data/ataparia/mmdet_lidar/work_dirs/second/epoch_100.pth"
    cfg.test_dataloader.dataset.data_root = sc["data_root"]
    cfg.test_dataloader.dataset.ann_file = sc["ann_file"]
    cfg.test_dataloader.dataset.default_cam_key = "CAM_FRONT"
    cfg.test_dataloader.dataset.modality = dict(use_lidar=True, use_camera=True)
    if "data_prefix" in cfg.test_dataloader.dataset:
        cfg.test_dataloader.dataset.data_prefix = dict(
            pts=sc["pts_prefix"], img=sc["img_prefix"])
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "LoadImageFromFile" and "data_root" in t:
            t["data_root"] = osp.join(sc["data_root"], sc["img_subdir"])
        if t.get("type") == "Pack3DDetInputs" and "img_path" in t.get("keys", []):
            t["keys"] = [k for k in t["keys"] if k != "img_path"]
            if "img_path" not in t.get("meta_keys", []):
                t["meta_keys"] = list(t.get("meta_keys", [])) + ["img_path"]
    model = init_model(cfg, MI.ENSEMBLE_CKPT, device="cuda:0")
    model.eval()
    return model, cfg


def build_vlc_dataset(sc):
    """Build a WaymoDataset (test pipeline) for the scenario, returning the
    dataset and original-sample_idx -> position map."""
    cfg = Config.fromfile(osp.join(PROJECT_ROOT, MI.VLC_CFG))
    cfg.test_data_root = sc["data_root"]
    ds_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    ds_cfg.data_root = sc["data_root"]
    ds_cfg.ann_file = sc["ann_file"]
    if "data_prefix" in ds_cfg:
        ds_cfg.data_prefix = dict(pts=sc["pts_prefix"], img=sc["img_prefix"])
    for t in ds_cfg.pipeline:
        if t.get("type") == "LoadImageFromFile" and "data_root" in t:
            t["data_root"] = osp.join(sc["data_root"], sc["img_subdir"])
    dataset = DATASETS.build(ds_cfg)
    with open(sc["pkl"], "rb") as f:
        data_list = pickle.load(f)["data_list"]
    assert len(data_list) == len(dataset), (
        f"pkl ({len(data_list)}) vs dataset ({len(dataset)}) length mismatch")
    sidx_to_pos = {d["sample_idx"]: i for i, d in enumerate(data_list)}
    return dataset, sidx_to_pos


def build_ensemble_dataset(sc, cfg):
    ds_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset = DATASETS.build(ds_cfg)
    with open(sc["pkl"], "rb") as f:
        data_list = pickle.load(f)["data_list"]
    assert len(data_list) == len(dataset), (
        f"pkl ({len(data_list)}) vs dataset ({len(dataset)}) length mismatch")
    sidx_to_pos = {d["sample_idx"]: i for i, d in enumerate(data_list)}
    return dataset, sidx_to_pos


def infer_frame(model, dataset, sidx_to_pos, sidx):
    """Return dict(bboxes_3d (N,7), scores (N,), labels (N,)) - UNFILTERED."""
    data = dataset[sidx_to_pos[sidx]]
    batch = pseudo_collate([data])
    with torch.no_grad():
        out = model.test_step(batch)
    pred = out[0].pred_instances_3d
    return dict(
        boxes_3d=pred.bboxes_3d.tensor.cpu().numpy().astype(np.float32),
        scores=pred.scores_3d.cpu().numpy().astype(np.float32),
        labels=pred.labels_3d.cpu().numpy().astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Clip selection
# ---------------------------------------------------------------------------
def _ctx_stats(frames, cond_dict):
    """Per-context summary used for ranking."""
    bns = [f["images"]["CAM_FRONT"]["img_path"] for f in frames]
    night_vals = [bool(cond_dict[bn][NIGHT_IDX0]) for bn in bns if bn in cond_dict]
    frac = float(np.mean(night_vals)) if night_vals else 0.0
    n_pc = 0
    for f in frames:
        for inst in f.get("instances", []):
            if inst["bbox_label_3d"] in (1, 2):
                n_pc += 1
    ts = np.array(sorted(f["timestamp"] for f in frames), dtype=np.int64)
    med_gap = int(np.median(np.diff(ts))) if len(ts) > 1 else 0
    return frac, n_pc, med_gap


def select_contexts(scenario, n=N_CLIPS):
    """Return a list of (context_name, frames_sorted_by_ts, frac, n_pc, med_gap)
    for the n chosen contexts of a scenario."""
    sc = SCENARIOS[scenario]
    cond_dict = _load_cond_jsonl(sc["cond_jsonl"])
    with open(sc["pkl"], "rb") as f:
        data_list = pickle.load(f)["data_list"]
    ctx = defaultdict(list)
    for d in data_list:
        ctx[d["context_name"]].append(d)

    cand = []
    for cname, frames in ctx.items():
        frames = sorted(frames, key=lambda d: d["timestamp"])
        frac, n_pc, med_gap = _ctx_stats(frames, cond_dict)
        cand.append(dict(name=cname, frames=frames, frac=frac,
                         n_pc=n_pc, med_gap=med_gap, n=len(frames)))

    # Scenario filter on day/night classification.
    if scenario == "night":
        cand = [c for c in cand if c["frac"] >= 0.5]
    elif scenario == "day":
        cand = [c for c in cand if c["frac"] < 0.5]
    # dawn_dusk: all contexts qualify (unseen split).

    sc_prefer_cont = sc["prefer_continuous"]

    def rank_key(c):
        # Prefer: has ped/cyc, then (for continuous sources) gap close to 100ms,
        # then more frames, then more ped/cyc instances.
        has_pc = 1 if c["n_pc"] > 0 else 0
        cont_pen = abs(c["med_gap"] - 100000) if sc_prefer_cont else c["med_gap"]
        return (-has_pc, cont_pen, -min(c["n"], WINDOW_LEN), -c["n_pc"])

    cand.sort(key=rank_key)
    chosen = cand[:n]
    return chosen, cond_dict


def window_for(ctx_entry, prefer_continuous):
    """Pick a contiguous window of up to WINDOW_LEN frames from a context.

    For continuous (dawn_dusk) sources, choose the WINDOW_LEN-frame window with
    the most pedestrian/cyclist GT instances. For subsampled sources just take
    the whole (short) context up to WINDOW_LEN.
    """
    frames = ctx_entry["frames"]
    if len(frames) <= WINDOW_LEN:
        return frames
    if not prefer_continuous:
        return frames[:WINDOW_LEN]
    # slide a window, score by #ped/cyc GT
    best_start, best_score = 0, -1
    pc = []
    for f in frames:
        c = sum(1 for inst in f.get("instances", []) if inst["bbox_label_3d"] in (1, 2))
        pc.append(c)
    pc = np.array(pc)
    csum = np.concatenate([[0], np.cumsum(pc)])
    for s in range(0, len(frames) - WINDOW_LEN + 1):
        score = int(csum[s + WINDOW_LEN] - csum[s])
        if score > best_score:
            best_score, best_start = score, s
    return frames[best_start:best_start + WINDOW_LEN]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
# pre-computed BGR colors (match the old matplotlib look)
_BEV_BG_BGR = (7, 7, 7)         # #070707
_BEV_PT_BGR = (176, 114, 76)    # #4C72B0 (seaborn/facebook blue)
_BEV_EGO_BGR = (40, 39, 214)    # #d62728


def render_lidar_bev(lidar_path, out_path):
    """Fast numpy rasterizer (no matplotlib). Same 880x880 pixel mapping as
    BEV_PARAMS so saved-frame overlays stay valid: u=(R-y)/(2R)*W, v=(R-x)/(2R)*H."""
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)[:, :3]
    x, y = pts[:, 0], pts[:, 1]
    m = (np.abs(x) < BEV_RANGE) & (np.abs(y) < BEV_RANGE)
    x, y = x[m], y[m]
    W = H = BEV_IMG
    R = BEV_RANGE
    u = np.clip(((R - y) / (2 * R) * W).astype(np.int32), 0, W - 1)
    v = np.clip(((R - x) / (2 * R) * H).astype(np.int32), 0, H - 1)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = _BEV_BG_BGR
    img[v, u] = _BEV_PT_BGR
    cv2.drawMarker(img, (W // 2, H // 2), _BEV_EGO_BGR,
                   cv2.MARKER_TRIANGLE_UP, 18, 3)
    cv2.imwrite(out_path, img)


def _draw_boxes(img_bgr, boxes, labels, lidar2img):
    # 3D boxes with a translucent shaded FRONT (heading) face so orientation is
    # readable on the 2D image (see tools/box_draw.py).
    return BD.draw_boxes_on_image(img_bgr, boxes, labels, lidar2img)


def _filter(pred, thr):
    keep = pred["scores"] >= thr
    return pred["boxes_3d"][keep], pred["labels"][keep]


def _even_scale_filter():
    return "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def encode_mp4(png_dir, out_mp4, fps=FPS):
    pat = osp.join(png_dir, "%05d.png")
    cmd = [FFMPEG, "-y", "-framerate", str(fps), "-i", pat,
           "-vf", _even_scale_filter(),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest():
    if osp.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"clips": []}


def save_manifest(man):
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def clip_done(scenario, nn):
    """A clip is 'done' if its 4 mp4 + data.pkl + info.json all exist."""
    d = osp.join(OUT_ROOT, scenario, f"clip_{nn:02d}")
    needed = ["rgb.mp4", "lidar_bev.mp4", "ssd_boxes_rgb.mp4",
              "vlc_boxes_rgb.mp4", "data.pkl", "info.json"]
    return all(osp.exists(osp.join(d, x)) for x in needed)


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------
def process_clip(scenario, nn, ctx_entry, cond_dict,
                 ssd_model, ssd_ds, ssd_map,
                 vlc_model, vlc_ds, vlc_map, idx_by_sample, sc):
    out_dir = osp.join(OUT_ROOT, scenario, f"clip_{nn:02d}")
    os.makedirs(out_dir, exist_ok=True)
    tmp_root = osp.join(out_dir, "_frames")
    streams = ["rgb", "lidar_bev", "ssd_boxes_rgb", "vlc_boxes_rgb"]
    png_dirs = {s: osp.join(tmp_root, s) for s in streams}
    for d in png_dirs.values():
        os.makedirs(d, exist_ok=True)

    window = window_for(ctx_entry, sc["prefer_continuous"])
    sample_indices = [d["sample_idx"] for d in window]
    n = len(sample_indices)

    img_dir = osp.join(sc["data_root"], sc["img_subdir"])
    lidar_dir = osp.join(sc["data_root"], sc["lidar_subdir"])

    frames_meta = []
    gt_counts = defaultdict(int)
    ssd_counts = defaultdict(int)
    vlc_counts = defaultdict(int)

    for i, sidx in enumerate(sample_indices):
        info = idx_by_sample[sidx]
        cf = info["images"]["CAM_FRONT"]
        bn = cf["img_path"]
        img_path = osp.join(img_dir, bn)
        lidar_path = osp.join(lidar_dir, info["lidar_points"]["lidar_path"])

        # calibration (promote 3x4 -> 4x4 where needed)
        def to4x4(m):
            m = np.array(m, dtype=np.float64)
            if m.shape == (3, 4):
                m = np.vstack([m, [0, 0, 0, 1]])
            return m
        cam2img = to4x4(cf["cam2img"])
        lidar2cam = to4x4(cf["lidar2cam"])
        lidar2img = to4x4(cf["lidar2img"])
        ego2global = np.array(info["ego2global"], dtype=np.float64)

        # conditions (10-d)
        conds = cond10(cond_dict[bn]) if bn in cond_dict else np.zeros(10, np.float32)

        # GT (classes 0/1/2 only)
        gt_boxes, gt_labels = [], []
        for inst in info.get("instances", []):
            lab = inst["bbox_label_3d"]
            if 0 <= lab <= 2:
                gt_boxes.append(np.array(inst["bbox_3d"], dtype=np.float32))
                gt_labels.append(int(lab))
                gt_counts[lab] += 1
        gt_boxes = np.stack(gt_boxes) if gt_boxes else np.zeros((0, 7), np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        # inference (unfiltered)
        ssd_pred = infer_frame(ssd_model, ssd_ds, ssd_map, sidx)
        vlc_pred = infer_frame(vlc_model, vlc_ds, vlc_map, sidx)

        # render
        fn = f"{i:05d}.png"
        img_bgr = cv2.imread(img_path)
        cv2.imwrite(osp.join(png_dirs["rgb"], fn), img_bgr)
        render_lidar_bev(lidar_path, osp.join(png_dirs["lidar_bev"], fn))
        s_boxes, s_labels = _filter(ssd_pred, VIDEO_SCORE_THR)
        v_boxes, v_labels = _filter(vlc_pred, VIDEO_SCORE_THR)
        cv2.imwrite(osp.join(png_dirs["ssd_boxes_rgb"], fn),
                    _draw_boxes(img_bgr, s_boxes, s_labels, lidar2img))
        cv2.imwrite(osp.join(png_dirs["vlc_boxes_rgb"], fn),
                    _draw_boxes(img_bgr, v_boxes, v_labels, lidar2img))

        for c in (0, 1, 2):
            ssd_counts[c] += int((s_labels == c).sum())
            vlc_counts[c] += int((v_labels == c).sum())

        frames_meta.append(dict(
            sample_idx=int(sidx),
            image_basename=bn,
            image_path=img_path,
            lidar_path=lidar_path,
            image_hw=(int(img_bgr.shape[0]), int(img_bgr.shape[1])),  # rgb video frame dims
            timestamp=int(info["timestamp"]),
            cam2img=cam2img,
            lidar2cam=lidar2cam,
            lidar2img=lidar2img,
            ego2global=ego2global,
            conditions=conds,
            gt=dict(boxes_3d=gt_boxes, labels=gt_labels),
            ssd=dict(boxes_3d=ssd_pred["boxes_3d"], scores=ssd_pred["scores"],
                     labels=ssd_pred["labels"]),
            vlc=dict(boxes_3d=vlc_pred["boxes_3d"], scores=vlc_pred["scores"],
                     labels=vlc_pred["labels"]),
        ))
        if i % 10 == 0 or i == n - 1:
            print(f"    [{scenario} clip_{nn:02d}] frame {i+1}/{n} sidx={sidx}", flush=True)

    # encode 4 mp4
    for s in streams:
        encode_mp4(png_dirs[s], osp.join(out_dir, f"{s}.mp4"))

    # nighttime fraction over the window
    night_vals = [bool(cond_dict[d["images"]["CAM_FRONT"]["img_path"]][NIGHT_IDX0])
                  for d in window if d["images"]["CAM_FRONT"]["img_path"] in cond_dict]
    night_frac = float(np.mean(night_vals)) if night_vals else 0.0
    ts = np.array([d["timestamp"] for d in window], dtype=np.int64)
    med_gap = int(np.median(np.diff(ts))) if len(ts) > 1 else 0

    # data.pkl
    data = dict(
        scenario=scenario,
        context_name=ctx_entry["name"],
        fps=FPS,
        video_score_threshold=VIDEO_SCORE_THR,
        # Self-contained: with these + the saved mp4s you can recreate ANY
        # modality x box-source overlay WITHOUT the raw Waymo data.
        #  - rgb / *_boxes_rgb frames: native res (per-frame image_hw); project
        #    3D boxes with per-frame lidar2img.
        #  - lidar_bev frames: 880x880; map boxes with bev_params (below).
        bev_params=BEV_PARAMS,
        class_names=CLASS_NAMES,
        class_colors_rgb=CLASS_COLORS,
        streams=["rgb", "lidar_bev", "ssd_boxes_rgb", "vlc_boxes_rgb"],
        frames=frames_meta,
    )
    with open(osp.join(out_dir, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    # info.json
    info_json = dict(
        scenario=scenario,
        context_name=ctx_entry["name"],
        clip=f"clip_{nn:02d}",
        n_frames=n,
        nighttime_fraction=round(night_frac, 4),
        fps=FPS,
        median_gap_us=med_gap,
        video_score_threshold=VIDEO_SCORE_THR,
        class_names=CLASS_NAMES,
        gt_counts={CLASS_NAMES[c]: int(gt_counts[c]) for c in (0, 1, 2)},
        ssd_pred_counts={CLASS_NAMES[c]: int(ssd_counts[c]) for c in (0, 1, 2)},
        vlc_pred_counts={CLASS_NAMES[c]: int(vlc_counts[c]) for c in (0, 1, 2)},
        sample_indices=[int(s) for s in sample_indices],
    )
    with open(osp.join(out_dir, "info.json"), "w") as f:
        json.dump(info_json, f, indent=2)

    # clean tmp frames
    subprocess.run(["rm", "-rf", tmp_root], check=False)

    print(f"  DONE {scenario} clip_{nn:02d}: {n} frames | night_frac={night_frac:.2f} "
          f"| GT V{gt_counts[0]}/P{gt_counts[1]}/C{gt_counts[2]} "
          f"| SSD V{ssd_counts[0]}/P{ssd_counts[1]}/C{ssd_counts[2]} "
          f"| VLC V{vlc_counts[0]}/P{vlc_counts[1]}/C{vlc_counts[2]}", flush=True)

    return dict(
        scenario=scenario,
        clip=f"clip_{nn:02d}",
        context_name=ctx_entry["name"],
        dir=out_dir,
        n_frames=n,
        nighttime_fraction=round(night_frac, 4),
        median_gap_us=med_gap,
        gt_counts=info_json["gt_counts"],
        ssd_pred_counts=info_json["ssd_pred_counts"],
        vlc_pred_counts=info_json["vlc_pred_counts"],
        mp4s={s: osp.join(out_dir, f"{s}.mp4") for s in streams},
        data_pkl=osp.join(out_dir, "data.pkl"),
        info_json=osp.join(out_dir, "info.json"),
        sample_indices=info_json["sample_indices"],
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(scenarios, n_clips, validate_only=False):
    os.makedirs(OUT_ROOT, exist_ok=True)
    man = load_manifest()
    done_keys = {(c["scenario"], c["clip"]) for c in man["clips"]}

    # Build VLC once (shared across all scenarios via prefix routing).
    print("Building VLC model (once, all condition dicts)...", flush=True)
    vlc_model = build_vlc_model()

    for scenario in scenarios:
        sc = SCENARIOS[scenario]
        print(f"\n===== SCENARIO: {scenario} =====", flush=True)
        chosen, cond_dict = select_contexts(scenario, n=n_clips)
        print(f"  selected {len(chosen)} contexts", flush=True)

        idx_by_sample = {d["sample_idx"]: d
                         for d in pickle.load(open(sc["pkl"], "rb"))["data_list"]}

        # Build ensemble + datasets for this split.
        print(f"  building Ensemble + datasets for {scenario}...", flush=True)
        ssd_model, ssd_cfg = build_ensemble_model(sc)
        ssd_ds, ssd_map = build_ensemble_dataset(sc, ssd_cfg)
        vlc_ds, vlc_map = build_vlc_dataset(sc)

        clips_to_do = range(1, (1 if validate_only else n_clips) + 1)
        for nn in clips_to_do:
            if nn > len(chosen):
                print(f"  WARN only {len(chosen)} contexts; skipping clip_{nn:02d}")
                break
            if (scenario, f"clip_{nn:02d}") in done_keys and clip_done(scenario, nn):
                print(f"  skip {scenario} clip_{nn:02d} (already done)")
                continue
            ctx_entry = chosen[nn - 1]
            print(f"  -> {scenario} clip_{nn:02d}: ctx={ctx_entry['name']} "
                  f"(frames={ctx_entry['n']}, frac={ctx_entry['frac']:.2f}, "
                  f"pc={ctx_entry['n_pc']}, gap={ctx_entry['med_gap']})", flush=True)
            rec = process_clip(scenario, nn, ctx_entry, cond_dict,
                               ssd_model, ssd_ds, ssd_map,
                               vlc_model, vlc_ds, vlc_map, idx_by_sample, sc)
            # update manifest incrementally
            man["clips"] = [c for c in man["clips"]
                            if not (c["scenario"] == scenario and c["clip"] == rec["clip"])]
            man["clips"].append(rec)
            save_manifest(man)

        # free ensemble + datasets before next split
        del ssd_model, ssd_ds, vlc_ds
        torch.cuda.empty_cache()

    print(f"\nALL DONE. Manifest: {MANIFEST_PATH}  ({len(man['clips'])} clips)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="only clip_01 of each scenario")
    ap.add_argument("--scenario", default=None,
                    choices=list(SCENARIOS.keys()),
                    help="run a single scenario")
    ap.add_argument("--n-clips", type=int, default=N_CLIPS)
    args = ap.parse_args()
    scs = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    run(scs, args.n_clips, validate_only=args.validate)
