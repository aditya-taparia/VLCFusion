"""Qualitative ATR fusion clips: one target track per clip, VLCFusion detections.

Each clip is a short video of ONE target vehicle at ONE range (a scenario segment tag,
e.g. cegr02003_0001), sampled across aspect angles. Two panels per frame:
    LEFT  = Visible input (raw i1co frame)          -- second modality, shown as-is
    RIGHT = IR input (cegr) + VLCFusion detections   -- boxes decode to IR coords
IR and VIS are NOT co-registered (~50 px offset), so boxes go on IR only; VIS is the
raw co-input.

Model / preprocessing mirror run_eval_ensemble.py exactly (validated by atr_smoke_test.py).
Distances: seen {1000,2000,3000,4000,5000} from *_seen (train+val+test), unseen
{1500,2500,3500,4500} from *_unseen (val+test). Frames of a tag are scattered across
splits (random per-frame split) -> gather by tag prefix, sort by frame number.

Run from inside "ATR Experiment" with the gen_ai env.
  CUDA_VISIBLE_DEVICES=0 python atr_clip_batch.py --one          # 1 seen clip, to preview
  CUDA_VISIBLE_DEVICES=0 python atr_clip_batch.py --split seen    # all 20 seen
  CUDA_VISIBLE_DEVICES=0 python atr_clip_batch.py                 # all 40
"""
import argparse, json, os, os.path as osp, pickle, subprocess, sys, collections
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from transformers import AutoConfig, AutoImageProcessor
from multimodal_detr import MultimodalDetr

DS = "/data/ataparia/Darpa_datasets/atr_dataset"
OUT_ROOT = "/data/ataparia/vlc_fusion_ECML/figures/atr_montage/clips"
BASE = "facebook/detr-resnet-50"
IR_BR, VIS_BR = "models/45k_seen_ir/checkpoint-24550", "models/45k_seen_vis/checkpoint-24550"
CKPT = "models/45k_vlcfusion_1blocks_7cond_lr2e5/model.safetensors"
NUM_VLC_BLOCKS = 1
COND_IDX = np.array([16, 13, 1, 11, 15, 19, 18])  # 1-based
IMG, THR, FPS, N_FRAMES = 480, 0.5, 12, 60
# The IR prediction sits a constant ~17 px BELOW the target in the 640x512 IR frame,
# measured flat across 1000-5000 m (a fixed decode/annotation offset, NOT target-relative:
# it is 0.7x box-height at 1000 m but 4x at 5000 m). Shift predictions up by this before
# mapping to the visible frame. Tune here if boxes still read low/high.
IR_BIAS_PY = 17.0
ID2LABEL = {0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
            5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20"}
DIST = {"02003": 1000, "02005": 1500, "02007": 2000, "02009": 2500, "02011": 3000,
        "02013": 3500, "02015": 4000, "02017": 4500, "02019": 5000}
SEEN_SCEN = ["02003", "02007", "02011", "02015", "02019"]
UNSEEN_SCEN = ["02005", "02009", "02013", "02017"]
SEEN_SPLITS = ["train", "val", "test"]
# Unseen conditions (vlm_*.json) exist only for `test`; using val frames would force a
# zero-vector fallback and make the conditioned prediction flicker. Test alone has
# 224-356 frames/tag -- plenty for 60-frame clips.
UNSEEN_SPLITS = ["test"]
BOX_BGR = (0, 215, 255)  # yellow, Fig-2 vehicle color


def frame_num(fn):
    return int(fn.split("_")[-1].split(".")[0])


def build_index(kind):
    """kind in {seen, unseen}. Return {tag: [frame dicts sorted by frame_num]}."""
    ir_dir, vis_dir = f"infrared_{kind}", f"visible_{kind}"
    splits = SEEN_SPLITS if kind == "seen" else UNSEEN_SPLITS
    by_tag = collections.defaultdict(list)
    for split in splits:
        irp = f"{DS}/{ir_dir}/{split}/metadata.jsonl"
        visp = f"{DS}/{vis_dir}/{split}/metadata.jsonl"
        if not osp.isfile(irp):
            continue
        vis = {m["image_id"]: m for m in (json.loads(l) for l in open(visp))}
        conds = json.load(open(f"conditions/{kind}/vlm_{split}.json"))
        for l in open(irp):
            m = json.loads(l)
            vm = vis.get(m["image_id"])
            if vm is None:
                continue
            fn = m["file_name"]
            tag = "_".join(fn.split("_")[:2])
            cvec = conds.get(str(m["image_id"]))
            cvec = ([1.0 if c else 0.0 for c in cvec] if cvec else None)
            gt = m["objects"]["bbox"][0] if m["objects"]["bbox"] else None
            vgt = vm["objects"]["bbox"][0] if vm["objects"]["bbox"] else None
            gtc = m["objects"]["category_name"][0] if m["objects"]["category_name"] else None
            by_tag[tag].append(dict(
                ir=f"{DS}/{ir_dir}/{split}/{fn}", vis=f"{DS}/{vis_dir}/{split}/{vm['file_name']}",
                fnum=frame_num(fn), cond=cvec, gt=gt, vgt=vgt, gtc=gtc,
                rng=(m["objects"]["range"][0] if m["objects"]["range"] else None)))
    for tag in by_tag:
        by_tag[tag].sort(key=lambda d: d["fnum"])
    return by_tag


def dominant_class(frames):
    c = collections.Counter(f["gtc"] for f in frames if f["gtc"])
    return c.most_common(1)[0][0] if c else "?"


def select_tags(by_tag, scenarios, per_scen):
    """Pick per_scen segments per scenario, favouring class variety across the selection."""
    chosen, used_cls = [], collections.Counter()
    for scen in scenarios:
        tags = sorted(t for t in by_tag if t[4:9] == scen and len(by_tag[t]) >= N_FRAMES)
        # rank this scenario's tags so rarer classes come first
        tags.sort(key=lambda t: (used_cls[dominant_class(by_tag[t])], t))
        for t in tags[:per_scen]:
            chosen.append(t); used_cls[dominant_class(by_tag[t])] += 1
    return chosen


def sample_window(frames, n=N_FRAMES):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


_TF = v2.Compose([v2.Resize((IMG, IMG)), v2.ToDtype(torch.float32, scale=True),
                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def preprocess(ir_pil, vis_pil):
    if ir_pil.size != vis_pil.size:
        ir_pil = ir_pil.resize(vis_pil.size)
    return torch.cat([_TF(tv_tensors.Image(ir_pil)), _TF(tv_tensors.Image(vis_pil))], 0)


def load_model():
    cfg = AutoConfig.from_pretrained(BASE, id2label=ID2LABEL,
                                     label2id={v: k for k, v in ID2LABEL.items()}, num_labels=10)
    model = MultimodalDetr(model_name_1=IR_BR, model_name_2=VIS_BR, config=cfg,
                           ensemble_method="VLCFusion", n_conditions=len(COND_IDX),
                           num_vlc_blocks=NUM_VLC_BLOCKS)
    from safetensors.torch import load_file
    res = model.load_state_dict(load_file(CKPT), strict=False, assign=True)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    return model.eval().cuda()


def infer(model, proc, frames, bs=16):
    """Return list of post-processed detection dicts (one per frame), IR-resolution xyxy."""
    out = []
    for i in range(0, len(frames), bs):
        chunk = frames[i:i + bs]
        px, labels, sizes = [], [], []
        for f in chunk:
            ir = Image.open(f["ir"]).convert("RGB")
            vis = Image.open(f["vis"]).convert("RGB")
            sizes.append((ir.height, ir.width))
            px.append(preprocess(ir, vis))
            cv = (torch.tensor(f["cond"])[torch.from_numpy(COND_IDX - 1)]
                  if f["cond"] else torch.zeros(len(COND_IDX)))
            # empty class_labels/boxes so DETR's matcher runs cleanly (no GT at inference)
            labels.append({"conditions": cv.cuda(),
                           "class_labels": torch.zeros(0, dtype=torch.long).cuda(),
                           "boxes": torch.zeros((0, 4)).cuda()})
        with torch.no_grad():
            o = model(pixel_values=torch.stack(px).cuda(), labels=labels)
        res = proc.post_process_object_detection(
            o, threshold=THR, target_sizes=torch.tensor(sizes))
        out.extend(res)
    return out


def ir_box_to_vis(pred_xyxy, ir_gt_xywh, vis_gt_xywh):
    """Map an IR-space detection into the visible frame via the paired GT correspondence.

    IR and VIS are co-registered only through the target: a per-frame similarity transform
    (centre shift + per-axis scale) that carries the IR GT box onto the VIS GT box. Exact
    when the prediction sits on the target (it does), so pred==ir_gt -> vis_gt.
    """
    igx, igy, igw, igh = ir_gt_xywh
    vgx, vgy, vgw, vgh = vis_gt_xywh
    ig_cx, ig_cy = igx + igw / 2, igy + igh / 2
    vg_cx, vg_cy = vgx + vgw / 2, vgy + vgh / 2
    sx, sy = vgw / max(igw, 1), vgh / max(igh, 1)
    px1, py1, px2, py2 = pred_xyxy
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    ncx, ncy = vg_cx + sx * (pcx - ig_cx), vg_cy + sy * (pcy - ig_cy)
    nw, nh = (px2 - px1) * sx, (py2 - py1) * sy
    return [ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2]


def draw_panel(frame, det, gt_class, dist, last_tf):
    """Visible-spectrum panel with the fusion detection mapped from IR. Returns (bgr, tf)."""
    vis = cv2.imread(frame["vis"])
    scale = 720 / vis.shape[0]
    vis = cv2.resize(vis, (int(vis.shape[1] * scale), 720))
    tf = (frame["gt"], frame["vgt"]) if (frame["gt"] and frame["vgt"]) else last_tf
    for s, l, b in zip(det["scores"], det["labels"], det["boxes"]):
        ib = [float(b[0]), float(b[1]) - IR_BIAS_PY, float(b[2]), float(b[3]) - IR_BIAS_PY]
        vb = ir_box_to_vis(ib, *tf) if tf else ib
        x1, y1, x2, y2 = [int(v * scale) for v in vb]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_BGR, 2)
        # predicted class + confidence, on a filled strip just above the box
        lab = f"{ID2LABEL[int(l)]} {float(s):.2f}"
        (lw, lh), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 - 4 if y1 - lh - 6 > 0 else y2 + lh + 6
        cv2.rectangle(vis, (x1, ly - lh - 4), (x1 + lw + 4, ly + 2), (0, 0, 0), -1)
        cv2.putText(vis, lab, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_BGR, 1, cv2.LINE_AA)
    # meaningful overlay: actual class + range
    txt = f"{gt_class}   {dist} m"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(vis, (8, 8), (18 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(vis, txt, (13, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis, tf


def encode(png_dir, n, out_mp4):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", osp.join(png_dir, "%04d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-movflags", "+faststart", out_mp4], check=True)


def render_clip(model, proc, tag, frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    win = sample_window(frames)
    dets = infer(model, proc, win)
    gt_class = dominant_class(win)
    dist = DIST.get(tag[4:9])
    fdir = osp.join(out_dir, "_frames"); os.makedirs(fdir, exist_ok=True)
    n_det, tf = 0, None
    for i, (f, d) in enumerate(zip(win, dets)):
        panel, tf = draw_panel(f, d, gt_class, dist, tf)
        cv2.imwrite(osp.join(fdir, f"{i:04d}.png"), panel)
        n_det += len(d["boxes"])
    encode(fdir, len(win), osp.join(out_dir, "fusion.mp4"))
    rng = np.median([f["rng"] for f in win if f["rng"]])
    info = dict(tag=tag, n_frames=len(win), gt_class=dominant_class(win),
                range_m=round(float(rng), 1), mean_det_per_frame=round(n_det / len(win), 2),
                dist_nominal=DIST.get(tag[4:9]))
    json.dump(info, open(osp.join(out_dir, "info.json"), "w"), indent=1)
    for p in os.listdir(fdir):
        os.remove(osp.join(fdir, p))
    os.rmdir(fdir)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="render a single seen clip and stop")
    ap.add_argument("--split", choices=["seen", "unseen"], help="only this split")
    args = ap.parse_args()

    model = load_model()
    proc = AutoImageProcessor.from_pretrained(BASE, do_resize=True,
                                              size={"max_height": IMG, "max_width": IMG})
    plan = []
    for kind, scen, per in [("seen", SEEN_SCEN, 4), ("unseen", UNSEEN_SCEN, 5)]:
        if args.split and args.split != kind:
            continue
        by_tag = build_index(kind)
        tags = select_tags(by_tag, scen, per)
        for t in tags:
            plan.append((kind, t, by_tag[t]))
        if args.one:
            plan = plan[:1]
            break

    print(f"rendering {len(plan)} clips")
    manifest = []
    for i, (kind, tag, frames) in enumerate(plan):
        nn = sum(1 for m in manifest if m["kind"] == kind) + 1
        out_dir = osp.join(OUT_ROOT, kind, f"clip_{nn:02d}")
        info = render_clip(model, proc, tag, frames, out_dir)
        info["kind"] = kind
        manifest.append(info)
        print(f"  [{i+1}/{len(plan)}] {kind}/clip_{nn:02d}  {tag}  {info['gt_class']:8s} "
              f"{info['dist_nominal']}m  det/frame {info['mean_det_per_frame']}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(manifest, open(osp.join(OUT_ROOT, "manifest.json"), "w"), indent=1)
    print(f"\nmanifest -> {OUT_ROOT}/manifest.json")


if __name__ == "__main__":
    main()
