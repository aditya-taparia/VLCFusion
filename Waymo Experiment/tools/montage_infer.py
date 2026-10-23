"""montage_infer.py

Run Fusion-SSD (Ensemble) and VLC-Fusion 3D detectors on chosen Waymo
dawn/dusk test frames and return pred_instances_3d.

This module does NOT modify any existing repo code/data. It builds each model
from (config, checkpoint) via mmdet3d.apis.init_model, constructs the per-frame
inputs by running each config's own test_pipeline, and calls model.test_step to
obtain predictions in LiDAR coordinates.

Path caveats handled via cfg-options style overrides applied in code:
  - data_root / ann_file resolve to /data/ataparia/LidarTraining/dawn_dusk-frames
  - VLC test_jsonl_file -> conditions/dawn_dusk_testing.jsonl
  - VLC test_data_root -> dawn_dusk-frames

Usage (standalone validation on one frame with Ensemble):
    CUDA_VISIBLE_DEVICES=0 python tools/montage_infer.py --validate 2000830
"""
import os
import os.path as osp
import sys
import argparse
import pickle
import copy

import numpy as np
import torch

PROJECT_ROOT = "/data/ataparia/VLCFusion/Waymo Experiment"
# Configs use custom_imports like 'configs.custom_ensemble.ensemble', which
# require PROJECT_ROOT on sys.path and to be the CWD (relative jsonl paths).
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate

# ---- Same module registrations as eval.py / trainer.py ----
from mmdet3d.utils import register_all_modules as register_mmdet3d
register_mmdet3d(init_default_scope=True)
from mmdet.utils import register_all_modules as register_mmdet
register_mmdet(init_default_scope=False)

import custom_waymo_metric  # noqa: F401  (registers CustomWaymoMetric)

# Patch MaxIoUAssigner to tolerate bboxes_3d/labels_3d gt (mirrors eval.py).
from mmdet.models.task_modules.assigners.max_iou_assigner import (
    MaxIoUAssigner as _MaxIoUAssigner)
_orig_assign = _MaxIoUAssigner.assign
def _patched_assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
    if not hasattr(gt_instances, 'bboxes') and hasattr(gt_instances, 'bboxes_3d'):
        gt_instances.bboxes = gt_instances.bboxes_3d
    if not hasattr(gt_instances, 'labels') and hasattr(gt_instances, 'labels_3d'):
        gt_instances.labels = gt_instances.labels_3d
    if (gt_instances_ignore is not None
            and not hasattr(gt_instances_ignore, 'bboxes')
            and hasattr(gt_instances_ignore, 'bboxes_3d')):
        gt_instances_ignore.bboxes = gt_instances_ignore.bboxes_3d
    return _orig_assign(self, pred_instances, gt_instances, gt_instances_ignore, **kwargs)
_MaxIoUAssigner.assign = _patched_assign

# Custom LoadImageFromFile that accepts data_root (mirrors eval.py).
from typing import Optional as _Optional
from mmcv.transforms import LoadImageFromFile as _LoadImageFromFile
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class LoadImageFromFile(_LoadImageFromFile):
    def __init__(self, data_root: _Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root

    def transform(self, results: dict) -> _Optional[dict]:
        if self.data_root is not None and 'img_path' in results:
            orig = results['img_path']
            if not osp.isabs(orig) or not osp.exists(orig):
                results['img_path'] = osp.join(self.data_root, osp.basename(orig))
        return super().transform(results)
# ---- End registrations ----

from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS  # noqa: F401  (ensure registry populated)
from mmdet3d.registry import MODELS as MMDET3D_MODELS
from mmdet.registry import MODELS as MMDET_MODELS

# The Ensemble config references mmdet losses (FocalLoss, SmoothL1Loss,
# CrossEntropyLoss) WITHOUT the 'mmdet.' scope prefix. When building the model
# via init_model (outside a Runner), the mmdet3d MODELS registry cannot resolve
# these. Alias them into the mmdet3d registry at runtime (no file changes).
for _name in ("FocalLoss", "SmoothL1Loss", "CrossEntropyLoss"):
    if _name not in MMDET3D_MODELS.module_dict and _name in MMDET_MODELS.module_dict:
        MMDET3D_MODELS.register_module(
            name=_name, module=MMDET_MODELS.module_dict[_name], force=True)


DATA_ROOT = "/data/ataparia/LidarTraining/dawn_dusk-frames/"
TEST_PKL = osp.join(DATA_ROOT, "waymo_infos_test.pkl")
CONDITIONS_JSONL = "conditions/dawn_dusk_testing.jsonl"

ENSEMBLE_CFG = "configs/ensemble.py"
ENSEMBLE_CKPT = "/data/ataparia/mmdet_lidar/work_dirs/ensemble/epoch_19.pth"

VLC_CFG = "configs/vlc_fusion_10_conditions_no_cosine.py"
# NOTE: the brief named the 'no_cosine' epoch_30 checkpoint as the headline, but
# that training run DIVERGED: every epoch >= ~21 in
#   YOUR_PATH/vlc_fusion_10_conditions_no_cosine/
# has NaN fusion weights (rgb_transform_layer / fusion_head / bbox_head) and
# produces ZERO detections. The sibling run with the same VLCFusion architecture
# (10 conditions) is clean at every epoch including epoch_30, so we use it:
#   YOUR_PATH/vlc_fusion_10_conditions/epoch_30.pth   (CLEAN, high-conf preds)
# Both share configs/vlc_fusion_10_conditions_no_cosine.py's model definition.
VLC_CKPT_NAN = ("/data/ataparia/VLCFusion/Waymo Experiment/YOUR_PATH/"
                "vlc_fusion_10_conditions_no_cosine/epoch_30.pth")  # broken (NaN)
VLC_CKPT = ("/data/ataparia/VLCFusion/Waymo Experiment/YOUR_PATH/"
            "vlc_fusion_10_conditions/epoch_30.pth")  # clean, used here


def _load_infos():
    with open(TEST_PKL, "rb") as f:
        info = pickle.load(f)
    data_list = info["data_list"]
    idx_by_sample = {d["sample_idx"]: d for d in data_list}
    return idx_by_sample


def _build_ensemble_cfg():
    cfg = Config.fromfile(osp.join(PROJECT_ROOT, ENSEMBLE_CFG))
    # Resolve stale /mnt paths to /data and point the LiDAR sub-detector path.
    cfg.model.lidar_model_path = "/data/ataparia/mmdet_lidar/work_dirs/second/epoch_100.pth"
    # Ensemble needs camera images too (RGB branch). Force test pipeline image root.
    cfg.test_dataloader.dataset.data_root = DATA_ROOT
    cfg.test_dataloader.dataset.ann_file = "waymo_infos_test.pkl"
    # WaymoDataset only promotes per-cam fields (img_path, cam2img, lidar2img)
    # to the top level when default_cam_key is set AND modality.use_camera=True.
    # The ensemble config omits the cam key and sets use_camera=False even
    # though the model uses an RGB branch; set both so LoadImageFromFile finds
    # 'img_path'. (Does not modify the on-disk config.)
    cfg.test_dataloader.dataset.default_cam_key = "CAM_FRONT"
    cfg.test_dataloader.dataset.modality = dict(use_lidar=True, use_camera=True)
    # Rewrite stale /mnt image root in the test pipeline -> dawn_dusk testing.
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "LoadImageFromFile" and "data_root" in t:
            t["data_root"] = DATA_ROOT + "testing/image_0"
        # The ensemble config lists 'img_path' in Pack3DDetInputs.keys, but it
        # is not a packable input key (it belongs in meta_keys, where it also
        # appears). Drop it from keys so Pack3DDetInputs does not raise.
        if t.get("type") == "Pack3DDetInputs" and "img_path" in t.get("keys", []):
            t["keys"] = [k for k in t["keys"] if k != "img_path"]
            if "img_path" not in t.get("meta_keys", []):
                t["meta_keys"] = list(t.get("meta_keys", [])) + ["img_path"]
    return cfg


def _build_vlc_cfg():
    cfg = Config.fromfile(osp.join(PROJECT_ROOT, VLC_CFG))
    cfg.model.lidar_model_path = "/data/ataparia/mmdet_lidar/work_dirs/second/epoch_100.pth"
    cfg.model.test_jsonl_file = CONDITIONS_JSONL
    # Use dawn_dusk split for the test data + image root.
    cfg.test_data_root = DATA_ROOT
    cfg.test_dataloader.dataset.data_root = DATA_ROOT
    cfg.test_dataloader.dataset.ann_file = "waymo_infos_test.pkl"
    # Rewrite LoadImageFromFile data_root in the test pipeline to dawn_dusk.
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "LoadImageFromFile" and "data_root" in t:
            t["data_root"] = DATA_ROOT + "testing/image_0"
    return cfg


def build_model(which):
    """which in {'ensemble','vlc'} -> (model, cfg)."""
    os.chdir(PROJECT_ROOT)  # ensure CWD for relative jsonl paths
    if which == "ensemble":
        cfg = _build_ensemble_cfg()
        ckpt = ENSEMBLE_CKPT
    elif which == "vlc":
        cfg = _build_vlc_cfg()
        ckpt = VLC_CKPT
    else:
        raise ValueError(which)
    model = init_model(cfg, ckpt, device="cuda:0")
    model.eval()
    return model, cfg


def _build_dataset(cfg):
    """Build the WaymoDataset from cfg.test_dataloader.dataset and a
    (original) sample_idx -> dataset-index lookup.

    NOTE: WaymoDataset overwrites each item's 'sample_idx' with the sequential
    dataset position, so the original Waymo sample_idx (e.g. 2000830) is lost
    inside the dataset. The pkl data_list order is preserved 1:1 with the
    dataset order, so we map original sample_idx -> position via the raw pkl.
    """
    ds_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dataset = DATASETS.build(ds_cfg)
    with open(TEST_PKL, "rb") as f:
        info = pickle.load(f)
    data_list = info["data_list"]
    assert len(data_list) == len(dataset), (
        f"pkl ({len(data_list)}) and dataset ({len(dataset)}) length mismatch")
    sidx_to_pos = {d["sample_idx"]: i for i, d in enumerate(data_list)}
    return dataset, sidx_to_pos


def run_inference(which, sample_indices, score_thr=0.3, model=None, cfg=None,
                  dataset=None, sidx_to_pos=None, verbose=True):
    """Return {sample_idx: dict(bboxes_3d, scores_3d, labels_3d)} filtered by score."""
    if model is None or cfg is None:
        model, cfg = build_model(which)
    if dataset is None or sidx_to_pos is None:
        dataset, sidx_to_pos = _build_dataset(cfg)

    results = {}
    for sidx in sample_indices:
        if sidx not in sidx_to_pos:
            raise KeyError(f"sample_idx {sidx} not found in dataset")
        data = dataset[sidx_to_pos[sidx]]
        batch = pseudo_collate([data])
        with torch.no_grad():
            out = model.test_step(batch)
        ds = out[0]
        pred = ds.pred_instances_3d
        bboxes = pred.bboxes_3d
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()
        keep = scores >= score_thr
        bboxes_t = bboxes.tensor.cpu().numpy()[keep]
        res = {
            "bboxes_3d": bboxes_t,           # (N,7) x,y,z,l,w,h,yaw (LiDAR)
            "bboxes_obj": bboxes[keep],      # LiDARInstance3DBoxes (for .corners)
            "scores_3d": scores[keep],
            "labels_3d": labels[keep],
        }
        results[sidx] = res
        if verbose:
            nper = {c: int((labels[keep] == c).sum()) for c in (0, 1, 2)}
            top = np.sort(scores[keep])[::-1][:5]
            print(f"[{which}] frame {sidx}: {keep.sum()} boxes "
                  f"(C={nper[0]} P={nper[1]} Cyc={nper[2]}) "
                  f"top scores={np.round(top,3).tolist()}")
    return results, model, cfg, dataset, sidx_to_pos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=None,
                    help="single sample_idx to validate with Ensemble")
    ap.add_argument("--which", default="ensemble", choices=["ensemble", "vlc"])
    ap.add_argument("--score_thr", type=float, default=0.3)
    args = ap.parse_args()

    if args.validate is not None:
        res = run_inference(args.which, [args.validate],
                            score_thr=args.score_thr)[0]
        r = res[args.validate]
        print("bboxes_3d shape:", r["bboxes_3d"].shape)
        print("first box:", r["bboxes_3d"][0] if len(r["bboxes_3d"]) else "NONE")
