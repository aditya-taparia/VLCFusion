"""Smoke test: load the VLCFusion ATR model, run ONE seen test frame, draw boxes.

Mirrors run_eval_ensemble.py's build/load/preprocess exactly, but for a single frame,
so we can confirm the model produces sane detections before building the clip pipeline.
Run from inside "ATR Experiment" with the gen_ai env.
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
from transformers import AutoConfig, AutoImageProcessor
from multimodal_detr import MultimodalDetr

BASE = "facebook/detr-resnet-50"
IR_BR = "models/45k_seen_ir/checkpoint-24550"
VIS_BR = "models/45k_seen_vis/checkpoint-24550"
CKPT = "models/45k_vlcfusion_1blocks_7cond_lr2e5/model.safetensors"
NUM_VLC_BLOCKS = 1
COND_IDX = np.array([16, 13, 1, 11, 15, 19, 18])  # 1-based
IMG = 480
ID2LABEL = {0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
            5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20"}
DS = "/data/ataparia/Darpa_datasets/atr_dataset"


def load_model():
    cfg = AutoConfig.from_pretrained(BASE, id2label=ID2LABEL,
                                     label2id={v: k for k, v in ID2LABEL.items()}, num_labels=10)
    model = MultimodalDetr(model_name_1=IR_BR, model_name_2=VIS_BR, config=cfg,
                           ensemble_method="VLCFusion", n_conditions=len(COND_IDX),
                           num_vlc_blocks=NUM_VLC_BLOCKS)
    from safetensors.torch import load_file
    sd = load_file(CKPT)
    res = model.load_state_dict(sd, strict=False, assign=True)
    print(f"loaded: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    return model.eval().cuda()


def preprocess(ir_pil, vis_pil):
    if ir_pil.size != vis_pil.size:
        ir_pil = ir_pil.resize(vis_pil.size)
    tf = v2.Compose([v2.Resize((IMG, IMG)), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ir = tf(tv_tensors.Image(ir_pil))
    vis = tf(tv_tensors.Image(vis_pil))
    return torch.cat([ir, vis], dim=0).unsqueeze(0)  # [1,6,480,480]


def main():
    # pick a seen test frame that has a GT box; VIS is paired by image_id (not filename)
    meta = [json.loads(l) for l in open(f"{DS}/infrared_seen/test/metadata.jsonl")]
    vmeta = {m["image_id"]: m for m in
             (json.loads(l) for l in open(f"{DS}/visible_seen/test/metadata.jsonl"))}
    conds = json.load(open("conditions/seen/vlm_test.json"))
    rec = next(m for m in meta if m["objects"]["bbox"])
    fn = rec["file_name"]
    vis_fn = vmeta[rec["image_id"]]["file_name"]
    print(f"frame IR {fn} <-> VIS {vis_fn} | GT {rec['objects']['category_name']} range {rec['objects']['range']}")

    ir_pil = Image.open(f"{DS}/infrared_seen/test/{fn}").convert("RGB")
    vis_pil = Image.open(f"{DS}/visible_seen/test/{vis_fn}").convert("RGB")
    W, H = ir_pil.size
    px = preprocess(ir_pil, vis_pil).cuda()

    cvec = conds.get(str(rec["image_id"]))
    cvec = (torch.tensor([1.0 if c else 0.0 for c in cvec])[torch.from_numpy(COND_IDX - 1)]
            if cvec else torch.zeros(len(COND_IDX)))
    labels = [{"conditions": cvec.cuda(), "class_labels": torch.tensor([], dtype=torch.long).cuda(),
               "boxes": torch.zeros((0, 4)).cuda()}]

    model = load_model()
    proc = AutoImageProcessor.from_pretrained(BASE, do_resize=True,
                                              size={"max_height": IMG, "max_width": IMG})
    with torch.no_grad():
        out = model(pixel_values=px, labels=labels)
    res = proc.post_process_object_detection(out, threshold=0.3,
                                             target_sizes=torch.tensor([[H, W]]))[0]
    print(f"\ndetections @0.3: {len(res['boxes'])}")
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        print(f"  {ID2LABEL[int(l)]:8s} {float(s):.2f}  box {[round(float(x)) for x in b]}")
    gt = rec["objects"]["bbox"][0]
    print(f"  GT box (xywh): {[round(v) for v in gt]} -> xyxy {[round(gt[0]),round(gt[1]),round(gt[0]+gt[2]),round(gt[1]+gt[3])]}")


if __name__ == "__main__":
    main()
