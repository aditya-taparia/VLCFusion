"""smoke_test_single_tfrecord.py — verify the Waymo preprocessing on ONE tfrecord.

Runs the exact same Waymo2KITTI conversion used by `lidar_dataset_creation.py`,
but on a single tfrecord (and only the first --num-frames frames), into a throwaway
output dir, then checks that images, point clouds, and the infos pkl were produced.

Usage:
  # real run (needs one downloaded perception v1.4.x tfrecord):
  python smoke_test_single_tfrecord.py --tfrecord /path/to/segment-XXXX.tfrecord

  # plumbing-only dry run (no data; validates imports + converter wiring):
  python smoke_test_single_tfrecord.py --dry-run

Run with the `waymo` env:
  /data/ataparia/conda/envs/waymo/bin/python smoke_test_single_tfrecord.py ...
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only; converter needs no GPU
# waymo_open_dataset calls ParseFromString(bytearray(...)), which protobuf>=4 (upb impl)
# rejects. Forcing the pure-python impl accepts it. NOTE: pure-python is SLOW for the
# full dataset — for production prefer `pip install 'protobuf==3.20.*'` (fast cpp impl,
# also accepts bytearray) and drop this line.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import os.path as osp
import sys
import argparse
import shutil
import tempfile

HERE = osp.dirname(osp.abspath(__file__))
EXP_ROOT = osp.dirname(HERE)                 # ".../Waymo Experiment"
for p in (HERE, EXP_ROOT):                    # so `import waymo_converter` resolves
    if p not in sys.path:
        sys.path.insert(0, p)


def read_frames(tfrecord_path, num_frames):
    import tensorflow as tf
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()
    from waymo_open_dataset import dataset_pb2 as open_dataset
    frames = []
    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for i, data in enumerate(ds):
        if i >= num_frames:
            break
        fr = open_dataset.Frame()
        fr.ParseFromString(data.numpy())  # bytes; newer protobuf rejects bytearray
        frames.append(fr)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", default=None, help="path to a perception v1.4.x tfrecord")
    ap.add_argument("--num-frames", type=int, default=1)
    ap.add_argument("--out", default=None, help="output dir (default: temp dir, auto-cleaned)")
    ap.add_argument("--split", default="testing", choices=["training", "validation", "testing"])
    ap.add_argument("--keep", action="store_true", help="keep the output dir")
    ap.add_argument("--dry-run", action="store_true", help="no tfrecord; validate wiring only")
    args = ap.parse_args()

    import waymo_converter as waymo  # noqa: E402  (after sys.path setup)
    print("[ok] imported waymo_converter; Waymo2KITTI present:", hasattr(waymo, "Waymo2KITTI"))

    if args.dry_run:
        frames = []
        print("[dry-run] using an empty frame list")
    else:
        if not args.tfrecord or not osp.isfile(args.tfrecord):
            sys.exit(f"[FAIL] --tfrecord not found: {args.tfrecord}")
        print(f"[..] reading first {args.num_frames} frame(s) from {args.tfrecord}")
        frames = read_frames(args.tfrecord, args.num_frames)
        print(f"[ok] parsed {len(frames)} frame(s)")
        if frames:
            ctx = frames[0].context
            print(f"     context: location={ctx.stats.location} "
                  f"time_of_day={ctx.stats.time_of_day} weather={ctx.stats.weather}")

    out_dir = args.out or tempfile.mkdtemp(prefix="waymo_smoke_")
    save_dir = osp.join(out_dir, args.split)
    prefix = {"training": "0", "validation": "1", "testing": "2"}[args.split]
    print(f"[..] converting -> {save_dir} (prefix={prefix}, workers=0)")

    converter = waymo.Waymo2KITTI(
        frames, save_dir, prefix=prefix, workers=0,
        test_mode=False, info_prefix="waymo", max_sweeps=1,
        split=args.split, save_senor_data=True, save_cam_instances=True,
    )
    converter.convert()

    # ---- verify outputs ----
    info_pkl = osp.join(out_dir, f"waymo_infos_{ {'training':'train','validation':'val','testing':'test'}[args.split] }.pkl")
    import mmengine
    checks = []
    pkl_ok = osp.isfile(info_pkl)
    checks.append((f"infos pkl exists ({osp.basename(info_pkl)})", pkl_ok))
    n_imgs = sum(len(os.listdir(f"{out_dir}/{args.split}/image_{i}"))
                 for i in range(5) if osp.isdir(f"{out_dir}/{args.split}/image_{i}"))
    n_bins = len(os.listdir(f"{out_dir}/{args.split}/velodyne")) if osp.isdir(f"{out_dir}/{args.split}/velodyne") else 0
    if not args.dry_run:
        checks.append(("camera images written", n_imgs > 0))
        checks.append(("lidar .bin written", n_bins > 0))

    n_infos = 0
    if pkl_ok:
        d = mmengine.load(info_pkl)
        n_infos = len(d.get("data_list", []))
        print(f"[info] pkl metainfo={d.get('metainfo')} data_list={n_infos}")
        if n_infos:
            keys = sorted(d["data_list"][0].keys())
            print(f"[info] first sample keys: {keys}")
        if not args.dry_run:
            checks.append(("infos data_list non-empty", n_infos > 0))

    print(f"[info] images={n_imgs} bins={n_bins}")
    print("\n==== RESULTS ====")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    if not args.keep:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"[cleanup] removed {out_dir}")
    else:
        print(f"[kept] {out_dir}")

    print("\nSMOKE TEST:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
