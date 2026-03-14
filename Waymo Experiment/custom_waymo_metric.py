"""Drop-in replacement for mmdet3d's WaymoMetric.

Fixes two issues in the upstream implementation:
  1. The eval binary is referenced via a relative path that only works when
     the working directory is the mmdet3d package root.
  2. Only the object-type level metrics are parsed; range-breakdown metrics
     from the binary output are silently discarded.

Usage:
  Set ``type='CustomWaymoMetric'`` in your config's val_evaluator /
  test_evaluator, and make sure this module is imported before the runner
  is built (so the registry picks it up).
"""

import os.path as osp
import re
import subprocess
from typing import Dict, Optional

from mmdet3d.evaluation.metrics.waymo_metric import WaymoMetric
from mmdet3d.registry import METRICS
from mmengine.logging import print_log


def _get_eval_bin(name: str) -> str:
    """Resolve the absolute path to a Waymo eval binary shipped with mmdet3d."""
    import mmdet3d.evaluation.functional.waymo_utils as _wu
    path = osp.join(osp.dirname(_wu.__file__), name)
    if not osp.isfile(path):
        raise FileNotFoundError(
            f'Waymo eval binary not found at {path}. '
            'Make sure mmdet3d is installed with the pre-built binaries.')
    return path


def _parse_mAP_output(ret_texts: str) -> Dict[str, float]:
    """Parse all lines from compute_detection_metrics_main output,
    including both object-type and range-breakdown metrics."""
    ap_dict: Dict[str, float] = {}
    for line in ret_texts.strip().splitlines():
        m = re.match(
            r'^(.+?):\s*\[mAP\s+([\d.]+(?:e[+-]?\d+)?)\]'
            r'\s*\[mAPH\s+([\d.]+(?:e[+-]?\d+)?)\]',
            line)
        if m:
            raw_key = m.group(1)
            ap_dict[f'{raw_key} mAP'] = float(m.group(2))
            ap_dict[f'{raw_key} mAPH'] = float(m.group(3))

    type_map = {
        'OBJECT_TYPE_TYPE_VEHICLE': 'Vehicle',
        'OBJECT_TYPE_TYPE_PEDESTRIAN': 'Pedestrian',
        'OBJECT_TYPE_TYPE_SIGN': 'Sign',
        'OBJECT_TYPE_TYPE_CYCLIST': 'Cyclist',
    }
    friendly: Dict[str, float] = {}
    for raw_key, val in ap_dict.items():
        nice = raw_key
        for prefix, short in type_map.items():
            nice = nice.replace(prefix, short)
        nice = nice.replace('_LEVEL_1', '/L1').replace('_LEVEL_2', '/L2')
        nice = nice.replace('RANGE_TYPE_', 'Range/')
        nice = nice.replace('_', ' ')
        friendly[nice] = val

    for level in ('L1', 'L2'):
        for metric in ('mAP', 'mAPH'):
            vals = []
            for cls in ('Vehicle', 'Pedestrian', 'Cyclist'):
                k = f'{cls}/{level} {metric}'
                if k in friendly:
                    vals.append(friendly[k])
            if vals:
                friendly[f'Overall/{level} {metric}'] = sum(vals) / len(vals)

    return friendly


def _parse_let_output(ret_texts: str) -> Dict[str, float]:
    """Parse all lines from compute_detection_let_metrics_main output."""
    ap_dict: Dict[str, float] = {}
    for line in ret_texts.strip().splitlines():
        m = re.match(
            r'^(.+?):\s*\[mAPL\s+([\d.]+(?:e[+-]?\d+)?)\]'
            r'\s*\[mAP\s+([\d.]+(?:e[+-]?\d+)?)\]'
            r'\s*\[mAPH\s+([\d.]+(?:e[+-]?\d+)?)\]',
            line)
        if m:
            raw_key = m.group(1).strip()
            ap_dict[f'{raw_key} mAPL'] = float(m.group(2))
            ap_dict[f'{raw_key} mAP'] = float(m.group(3))
            ap_dict[f'{raw_key} mAPH'] = float(m.group(4))

    for metric in ('mAPL', 'mAP', 'mAPH'):
        vals = []
        for cls in ('Vehicle', 'Pedestrian', 'Cyclist'):
            k = f'{cls} {metric}'
            if k in ap_dict:
                vals.append(ap_dict[k])
        if vals:
            ap_dict[f'Overall {metric}'] = sum(vals) / len(vals)
    return ap_dict


@METRICS.register_module(force=True)
class CustomWaymoMetric(WaymoMetric):
    """Drop-in replacement for WaymoMetric that resolves eval binary paths
    correctly regardless of the working directory, and parses the full output
    including per-range breakdowns."""

    def waymo_evaluate(
        self,
        result_prefix: str,
        metric: Optional[str] = None,
        logger=None,
    ) -> Dict[str, float]:
        if metric == 'mAP':
            eval_bin = _get_eval_bin('compute_detection_metrics_main')
            eval_str = f'{eval_bin} {result_prefix}.bin {self.waymo_bin_file}'
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts, logger=logger)
            return _parse_mAP_output(ret_texts)

        elif metric == 'LET_mAP':
            eval_bin = _get_eval_bin('compute_detection_let_metrics_main')
            eval_str = f'{eval_bin} {result_prefix}.bin {self.waymo_bin_file}'
            print(eval_str)
            ret_bytes = subprocess.check_output(eval_str, shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts, logger=logger)
            return _parse_let_output(ret_texts)

        return {}
