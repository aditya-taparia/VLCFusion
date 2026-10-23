"""overlay_from_video.py — recreate ANY modality x box-source overlay for a clip
using ONLY the saved videos + data.pkl (no raw Waymo data needed).

It pulls frame N straight out of the saved mp4 (rgb.mp4 or lidar_bev.mp4) and
draws GT / Fusion-SSD / VLC-Fusion 3D boxes from data.pkl:
  - rgb:        project 3D boxes with per-frame lidar2img (front-face shaded).
  - lidar_bev:  map 3D boxes to BEV pixels via data['bev_params'].

Examples:
  python tools/overlay_from_video.py CLIP_DIR --frame 30 --modality rgb       --source vlc
  python tools/overlay_from_video.py CLIP_DIR --frame 30 --modality lidar_bev --source vlc
"""
import os, os.path as osp, sys, pickle, argparse
import numpy as np
import cv2

PROJECT_ROOT = "/data/ataparia/VLCFusion/Waymo Experiment"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import tools.box_draw as BD           # self-contained geometry + box drawing

CLASS_COLORS = BD.CLASS_COLORS        # 0 yellow / 1 orange / 2 magenta (RGB 0-1)


def grab_frame(mp4_path, n):
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {n} from {mp4_path}")
    return frame  # BGR


def boxes_for(frame_rec, source, score_thr):
    if source == "gt":
        return frame_rec["gt"]["boxes_3d"], frame_rec["gt"]["labels"]
    pred = frame_rec[source]
    keep = pred["scores"] >= score_thr
    return pred["boxes_3d"][keep], pred["labels"][keep]


def draw_bev(bev_bgr, boxes, labels, bev):
    """Draw boxes onto a saved BEV frame using the constant bev_params transform."""
    R = bev["range_m"]; W = bev["img_w"]; H = bev["img_h"]
    img = bev_bgr.copy()
    if (img.shape[1], img.shape[0]) != (W, H):
        img = cv2.resize(img, (W, H))

    def to_px(x, y):
        u = (R - y) / (2 * R) * W
        v = (R - x) / (2 * R) * H
        return [u, v]

    for box, lab in zip(boxes, labels):
        c = BD.box_corners_3d(box)[:4, :2]           # bottom 4 corners (x,y)
        quad = np.array([to_px(px, py) for px, py in c], dtype=np.int32)
        col_rgb = CLASS_COLORS.get(int(lab), (1, 1, 1))
        col_bgr = tuple(int(255 * v) for v in col_rgb[::-1])
        cv2.polylines(img, [quad], True, col_bgr, 2, cv2.LINE_AA)
        # heading: center -> front-edge midpoint (corners 0,1 are the +x face)
        cen = c.mean(axis=0)
        frontmid = (c[0] + c[1]) / 2.0
        p0 = to_px(cen[0], cen[1]); p1 = to_px(frontmid[0], frontmid[1])
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                 col_bgr, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--modality", choices=["rgb", "lidar_bev"], default="rgb")
    ap.add_argument("--source", choices=["gt", "ssd", "vlc"], default="vlc")
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = pickle.load(open(osp.join(args.clip_dir, "data.pkl"), "rb"))
    fr = data["frames"][args.frame]
    boxes, labels = boxes_for(fr, args.source, args.score)

    if args.modality == "rgb":
        base = grab_frame(osp.join(args.clip_dir, "rgb.mp4"), args.frame)
        out = BD.draw_boxes_on_image(base, boxes, labels, np.array(fr["lidar2img"]))
    else:
        base = grab_frame(osp.join(args.clip_dir, "lidar_bev.mp4"), args.frame)
        out = draw_bev(base, boxes, labels, data["bev_params"])

    out_path = args.out or osp.join(
        args.clip_dir, f"_overlay_{args.modality}_{args.source}_f{args.frame:03d}.png")
    cv2.imwrite(out_path, out)
    print("wrote", out_path, f"({len(boxes)} {args.source} boxes, no raw data used)")


if __name__ == "__main__":
    main()
