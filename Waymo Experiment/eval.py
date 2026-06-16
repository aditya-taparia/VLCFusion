"""
Evaluation script for trained Waymo 3D detection models.

Usage:
    # Evaluate on night-day validation set:
    python eval.py --config configs/vlc_fusion_10_conditions.py \
                   --checkpoint YOUR_PATH/vlc_fusion_10_conditions/epoch_40.pth \
                   --split val

    # Evaluate on dawn-dusk test set:
    python eval.py --config configs/vlc_fusion_10_conditions.py \
                   --checkpoint YOUR_PATH/vlc_fusion_10_conditions/epoch_40.pth \
                   --split test

    # Evaluate both splits:
    python eval.py --config configs/vlc_fusion_10_conditions.py \
                   --checkpoint YOUR_PATH/vlc_fusion_10_conditions/epoch_40.pth \
                   --split both
"""
import argparse
import os
import os.path as osp

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# ---------- Same registrations as trainer.py ----------
from mmdet3d.utils import register_all_modules as register_mmdet3d
register_mmdet3d(init_default_scope=True)
from mmdet.utils import register_all_modules as register_mmdet
register_mmdet(init_default_scope=False)

from custom_waymo_metric import CustomWaymoMetric  # noqa: F401

from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner as _MaxIoUAssigner
_orig_assign = _MaxIoUAssigner.assign
def _patched_assign(self, pred_instances, gt_instances, gt_instances_ignore=None, **kwargs):
    if not hasattr(gt_instances, 'bboxes') and hasattr(gt_instances, 'bboxes_3d'):
        gt_instances.bboxes = gt_instances.bboxes_3d
    if not hasattr(gt_instances, 'labels') and hasattr(gt_instances, 'labels_3d'):
        gt_instances.labels = gt_instances.labels_3d
    if gt_instances_ignore is not None and not hasattr(gt_instances_ignore, 'bboxes') and hasattr(gt_instances_ignore, 'bboxes_3d'):
        gt_instances_ignore.bboxes = gt_instances_ignore.bboxes_3d
    return _orig_assign(self, pred_instances, gt_instances, gt_instances_ignore, **kwargs)
_MaxIoUAssigner.assign = _patched_assign

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
# ---------- End registrations ----------


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained 3D detection model on Waymo.')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--split', default='val', choices=['val', 'test', 'both'],
                        help='Which split(s) to evaluate')
    parser.add_argument('--work-dir', default=None,
                        help='Directory to save logs and output files')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default=None,
                        help='Override config settings, e.g. '
                             'model.test_jsonl_file=conditions/smolvlm_conditions/dawn_dusk_testing.jsonl')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir or cfg.get('work_dir', None) or 'eval_output'
    cfg.work_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)

    cfg.load_from = args.checkpoint
    cfg.launcher = 'none'

    splits = ['val', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        print(f'\n{"="*60}')
        print(f'  Running evaluation on {split} split...')
        print(f'{"="*60}\n')

        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        if split == 'val':
            runner.val()
        else:
            runner.test()


if __name__ == '__main__':
    main()
