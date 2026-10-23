# Waymo Official Evaluator — setup & usage

VLCFusion reports Waymo **3D mAP / mAPH (L1, L2)** using the *official* Waymo Open
Dataset metric binary, `compute_detection_metrics_main`. This is the binary that
produced the paper's Waymo numbers. It's a compiled C++ tool (built from
`waymo-open-dataset` with Bazel), invoked via `subprocess` in
[`custom_waymo_metric.py`](custom_waymo_metric.py).

## What ships in this repo

A pre-built, portable binary is **vendored and git-tracked** at:

```
Waymo Experiment/mmdet3d/evaluation/functional/waymo_utils/compute_detection_metrics_main
```

- ELF 64-bit, dynamically links only standard system libs (`libstdc++`, `libc`, …) —
  no TensorFlow/abseil runtime needed. Runs on most modern x86-64 Linux as-is.
- Sanity check: `./.../compute_detection_metrics_main` should print
  `Usage: ... prediction_filename ground_truth_filename.`

> The **LET** variant (`compute_detection_let_metrics_main`, used only for the
> optional `LET_mAP` metric) is **not** vendored — the paper uses mAP/mAPH only.
> Build it the same way (see below) if you need LET.

## How the code finds it

`custom_waymo_metric._get_eval_bin()` looks for the binary in this order:
1. the **installed** `mmdet3d` package (`mmdet3d/evaluation/functional/waymo_utils/`),
2. the **repo-vendored** copy above (fallback).

So evaluation works out-of-the-box when you run from this repo. To make it work for
code that imports mmdet3d from elsewhere, copy the binary into the installed package:

```bash
DEST=$(python -c 'import mmdet3d.evaluation.functional.waymo_utils as w, os; print(os.path.dirname(w.__file__))')
cp "Waymo Experiment/mmdet3d/evaluation/functional/waymo_utils/compute_detection_metrics_main" "$DEST/"
chmod +x "$DEST/compute_detection_metrics_main"
```

## Rebuilding from source (only if the vendored binary won't run on your platform)

```bash
git clone https://github.com/waymo-research/waymo-open-dataset.git
cd waymo-open-dataset/src
bazel build -c opt //waymo_open_dataset/metrics/tools:compute_detection_metrics_main
# (for LET:)  bazel build -c opt //waymo_open_dataset/metrics/tools:compute_detection_let_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
   "<path-to>/Waymo Experiment/mmdet3d/evaluation/functional/waymo_utils/"
```
Requires Bazel and the waymo-open-dataset build toolchain.

## Running an evaluation

1. **Ground-truth `.bin`** — build once per split with
   [`gt_bin_creator.py`](gt_bin_creator.py) (produces e.g. `gt_<variation>_<split>.bin`).
2. **Predictions `.bin`** — produced during eval (`eval.py` /
   `CustomWaymoMetric`), then scored as:
   ```
   compute_detection_metrics_main  <pred>.bin  <gt>.bin
   ```
   `custom_waymo_metric.py` parses the stdout into per-class L1/L2 mAP & mAPH
   (plus range breakdowns).
