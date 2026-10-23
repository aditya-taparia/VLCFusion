"""Render ONE 1000 m clip introducing the DATASET / TASK: both modalities side by side
[Visible | Infrared] of the same target, no model, no boxes -- just the two co-registered
sensor views with a class + distance caption. Random class unless a tag is given.

  CUDA_VISIBLE_DEVICES=0 python atr_dual_render.py           # random 1000 m class
  python atr_dual_render.py cegr02003_0011                   # a specific tag (T72)
  python atr_dual_render.py cegr02003_0011 --pred            # also draw VLCFusion detections
"""
import os, os.path as osp, sys, subprocess
import numpy as np
import cv2
from atr_clip_batch import (build_index, sample_window, dominant_class,
                            ir_box_to_vis, IR_BIAS_PY, ID2LABEL, BOX_BGR, FPS, IMG)

OUT_DIR = "/data/ataparia/vlc_fusion_ECML/figures/atr_montage/dual_examples"
SCEN_1000 = "02003"
PANEL_H = 512


def panel(img_bgr, tag_text):
    h = PANEL_H
    out = cv2.resize(img_bgr, (int(img_bgr.shape[1] * h / img_bgr.shape[0]), h))
    cv2.rectangle(out, (0, 0), (110, 30), (0, 0, 0), -1)
    cv2.putText(out, tag_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def label_box(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_BGR, 2)
    (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = y1 - 4 if y1 - lh - 6 > 0 else y2 + lh + 6
    cv2.rectangle(img, (x1, ly - lh - 4), (x1 + lw + 4, ly + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_BGR, 1, cv2.LINE_AA)


def draw_dual(frame, det, gt_class, dist, tf):
    ir = cv2.imread(frame["ir"]); vis = cv2.imread(frame["vis"])
    tf = (frame["gt"], frame["vgt"]) if (frame["gt"] and frame["vgt"]) else tf
    s_ir, s_vis = PANEL_H / ir.shape[0], PANEL_H / vis.shape[0]
    ir_p = cv2.resize(ir, (int(ir.shape[1] * s_ir), PANEL_H))
    vis_p = cv2.resize(vis, (int(vis.shape[1] * s_vis), PANEL_H))
    if det is not None:  # optional --pred overlay; intro mode draws nothing
        for s, l, b in zip(det["scores"], det["labels"], det["boxes"]):
            txt = f"{ID2LABEL[int(l)]} {float(s):.2f}"
            ib = [float(b[0]), float(b[1]) - IR_BIAS_PY, float(b[2]), float(b[3]) - IR_BIAS_PY]
            x1, y1, x2, y2 = [int(v * s_ir) for v in ib]                # IR panel: native coords
            label_box(ir_p, x1, y1, x2, y2, txt)
            vb = ir_box_to_vis(ib, *tf) if tf else ib                    # Visible panel: mapped
            x1, y1, x2, y2 = [int(v * s_vis) for v in vb]
            label_box(vis_p, x1, y1, x2, y2, txt)
    vis_p = panel(vis_p, "Visible"); ir_p = panel(ir_p, "Infrared")
    pad = np.full((PANEL_H, 6, 3), 40, np.uint8)
    combo = np.hstack([vis_p, pad, ir_p])
    ov = f"{gt_class}   {dist} m"
    (tw, th), _ = cv2.getTextSize(ov, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cx = (combo.shape[1] - tw) // 2
    cv2.rectangle(combo, (cx - 8, combo.shape[0] - th - 20), (cx + tw + 8, combo.shape[0] - 6), (0, 0, 0), -1)
    cv2.putText(combo, ov, (cx, combo.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return combo, tf


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_pred = "--pred" in sys.argv
    by_tag = build_index("seen")
    tags = sorted(t for t in by_tag if t[4:9] == SCEN_1000 and len(by_tag[t]) >= 60)
    tag = args[0] if args else str(np.random.default_rng().choice(tags))
    frames = by_tag[tag]
    gt_class = dominant_class(frames)
    print(f"tag {tag}  class {gt_class}  1000 m  ({len(frames)} frames)  pred={show_pred}")

    win = sample_window(frames)
    dets = [None] * len(win)
    if show_pred:
        from transformers import AutoImageProcessor
        from atr_clip_batch import load_model, infer
        model = load_model()
        proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", do_resize=True,
                                                  size={"max_height": IMG, "max_width": IMG})
        dets = infer(model, proc, win)
    os.makedirs(OUT_DIR, exist_ok=True)
    fdir = osp.join(OUT_DIR, "_frames"); os.makedirs(fdir, exist_ok=True)
    tf = None
    for i, (f, d) in enumerate(zip(win, dets)):
        combo, tf = draw_dual(f, d, gt_class, 1000, tf)
        cv2.imwrite(osp.join(fdir, f"{i:04d}.png"), combo)
    suffix = "_pred" if show_pred else ""
    out = osp.join(OUT_DIR, f"dual_1000m_{gt_class}{suffix}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", osp.join(fdir, "%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out], check=True)
    for p in os.listdir(fdir):
        os.remove(osp.join(fdir, p))
    os.rmdir(fdir)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
