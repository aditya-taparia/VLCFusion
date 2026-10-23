# Repository audit — 2026-10-23

Scope: local working tree, tracked-file inventory, README commands, selected data interfaces, lightweight runtime checks, and prediction-media provenance. This is not an end-to-end reproduction of the paper. Model/preprocessing fixes are deferred; this pass updates documentation and adds README media.

## What is present

- All **236 tracked Python files** exist. There were also **13 untracked Python files**, for 249 source files at the start of the audit.
- ATR's final `MultimodalDetr` / `VLCFusion` implementation, retained baselines, fusion trainer, evaluation scripts, and preprocessing helpers are present.
- Waymo's `VLCFusion` detector, experiment configs, trainer, evaluator, converters, and official `compute_detection_metrics_main` executable are present. Referenced config `_base_` files resolve.
- Main condition files and small-VLM alternatives are present. The four ATR VLM split files contain uniform 19-element vectors; the Waymo question list has 40 entries.
- Historical variants are documented in `DEVELOPMENT_HISTORY.md`; the `pre-release-archive` tag exists. Their intentional removal should not be confused with missing final-model code.

## Release files still outside Git

At the start of this audit, **44 files were untracked**. A fresh clone of the current commit will not contain them. The README and media added in this pass also need to be included when the update is committed.

| Group | Existing untracked paths |
| --- | --- |
| ATR environment and visualization | `ATR Experiment/requirements.txt`, `requirements.lock.txt`, `atr_smoke_test.py`, `atr_clip_batch.py`, `atr_dual_render.py` |
| Waymo environment/evaluator docs | `Waymo Experiment/requirements.txt`, `requirements.lock.txt`, `WAYMO_EVAL_SETUP.md` |
| Waymo preprocessing | `Waymo Experiment/mmdet_data_creation/smoke_test_single_tfrecord.py`; both text files in `data_creation/used_segments/` |
| Waymo visualization | `Waymo Experiment/tools/{box_draw,clip_batch,clip_render,montage_infer,montage_render,overlay_from_video}.py` |
| Figures | All 27 files under `plots/`: three Python scripts, two notebooks, and PDF/PNG outputs |

Five tracked root-level artifacts appear deleted: four radar PDFs and `plots.ipynb`. SHA-256 comparisons confirm identical copies under `plots/`; the notebook is now `plots/legacy_plots.ipynb`. The files have not been lost, but the moves are not recorded in Git yet.

Pre-existing edits to `ATR Experiment/ensemble_trainer.py`, `Waymo Experiment/custom_waymo_metric.py`, and root `requirements.txt` were preserved. Nothing was committed or pushed during the audit.

## Fixes needed before clean reproduction

### 1. ATR preprocessing does not yet match the trainer's data contract

In [`build_image_datasets.py`](../ATR%20Experiment/ATR%20data%20preprocessing/build_image_datasets.py):

- Images are saved under `<split>/images/`, but metadata `file_name` contains only the basename. The `imagefolder` loader and condition generator resolve it relative to `<split>/`, yielding a nonexistent path. A synthetic dataset reproduced the resulting `FileNotFoundError`.
- Metadata writes `objects.category_id`. [`ensemble_trainer.py`](../ATR%20Experiment/ensemble_trainer.py) reads `objects.category`. A synthetic preprocessing call produced one bounding box and zero class labels from a builder-style record.
- IR and visible IDs are allocated independently after skipped rows. A failure in one modality can shift subsequent IDs and pairing. Allocate a stable synchronized-pair ID and validate it before concatenating modalities.
- The helper's video-tag split is not a full implementation of the paper's seen/unseen distance split. Include explicit split manifests/recipes and ensure regenerated condition IDs correspond to those splits.

### 2. Newly generated Waymo conditions do not match the fusion loader

[`get_conditions.py`](../Waymo%20Experiment/get_conditions.py) writes `original_img_path`, but [`vlc_fusion.py`](../Waymo%20Experiment/configs/cbam_ensemble_vlm/vlc_fusion.py) accesses `data['image_path']`. Directly loading new generator output therefore raises `KeyError`. Bundled JSONL files already use `image_path`. Standardize the schema and validate condition lengths and sample coverage.

### 3. Dependency files are development snapshots, not portable lockfiles

- Root `requirements.txt` contains only comments and installs nothing.
- The ATR lock snapshot includes `file:///home/conda/feedstock_root/...` references and an editable dependency from an unrelated repository. These cannot serve as a portable environment recipe.
- ATR's short requirements list omits direct dependencies used in the documented workflow, including `timm`, Pandas, scikit-learn, and Matplotlib. Some are present transitively or in the larger snapshot.
- The Waymo short list omits the Transformers/timm stack imported by the fusion detector. Its lock snapshot contains those packages but omits `openai`, which is needed for condition generation. `get_conditions.py --help` also fails in the existing Waymo environment because `openai` is not installed.
- `pip check` in the existing Waymo environment reports four conflicts: `nuscenes-devkit` versus Matplotlib, `open3d` versus Pillow, OpenCV versus NumPy, and `openxlab` versus setuptools. In particular, the listed OpenCV 4.13 build requires NumPy >= 2 on this Python version, while Waymo pins NumPy 1.23.0.
- The existing ATR environment passes `pip check`; that does not establish that its exported snapshot can be installed on a new machine.

Create minimal, tested environment recipes independently for each experiment; preserve the original freezes as provenance if useful.

### 4. Waymo training and paths need a portable entry point

[`trainer.py`](../Waymo%20Experiment/trainer.py) imports `argparse` but never parses command-line arguments. It selects the no-cosine config, work directory, automatic resume, and GPU 1 through hard-coded constants, then starts training at module scope. The old README's positional config and `--work-dir` arguments had no effect.

The generic `tools/train.py` does accept CLI options, but lacks the experiment trainer's custom metric registration, image-loader override, and assigner patch. Do not substitute it without preserving those registrations.

Configs contain author-specific dataset and component-checkpoint paths. Updating only the top-level `data_root` after config evaluation does not update derived nested paths. Full TFRecord conversion and GT-bin creation also contain hard-coded paths, variation choices, and split logic. Rendering tools contain additional absolute project/output paths.

### 5. Required external artifacts and training recipes are not distributed

A clean clone still needs:

- Raw or prepared ATR/Waymo datasets and exact experiment split definitions.
- Pretrained single-modality detector components and fusion checkpoints, with download locations, architecture/condition settings, and checksums.
- A documented recipe or linked source for training the ATR single-modality components and Waymo RGB component; those stages are not supplied as complete dedicated training workflows here.
- A correspondence between each published result, config, checkpoint, condition source, and evaluation split. For example, the cosine Waymo config currently selects SmolVLM dawn/dusk conditions, while the no-cosine config selects main night/day conditions.

The bundled metric GT files are useful only for their matching frame selections. The optional LET metric executable is absent, as already documented in `WAYMO_EVAL_SETUP.md`.

### 6. Clarify visualization provenance before using clips as localization evidence

The README GIFs preserve existing recorded overlays. The ATR renderer's `draw_panel` applies a fixed vertical offset to IR detections, then uses paired ground-truth box correspondences to project them into the visible image. Its module-level description still describes an older two-panel output, while the current function writes a single visible panel.

These projected overlays can illustrate the task, but they should not be presented as raw visible-space model predictions or an independent measure of localization accuracy. A future demo should show native IR-coordinate predictions beside the visible input and document any calibration. The media manifest discloses the existing display transform.

ATR seen clips gather frames across train/validation/test. Their “seen” label denotes the distance regime; it does not certify that the clip is exclusively held-out test data. Unseen clips use test frames. The raw paired ATR recordings contain no predictions; the current README gallery uses recorded prediction clips throughout.

### 7. Remaining documentation/release items

- Add the YouTube talk URL when supplied; the README currently says “Talk (coming soon)”.
- No root `LICENSE` file is present; the maintainers still need to choose and document the repository license.
- Include the existing untracked source/dependency files and plot moves in the release commit.
- Add a small end-to-end data-contract check when fixing the preprocessing interfaces, then verify training/evaluation with real sample data and a checkpoint.

## Checks performed

| Check | Result |
| --- | --- |
| Parse all 249 local Python sources | Passed |
| Read all existing notebooks as JSON | Passed |
| Parse ATR condition JSON and all 44 Waymo condition JSONL files | Passed |
| Resolve config `_base_` file references | Passed |
| ATR preprocessing, condition generator, trainer, and ensemble evaluator `--help` | Passed in the existing ATR environment |
| ATR fusion feature/query layers: CPU forward and backward | Passed; expected output shapes and finite input gradients |
| Waymo `eval.py --help` and both final VLC config imports | Passed in the existing Waymo environment |
| Waymo single-TFRecord converter `--dry-run` | Passed; an empty infos file was created and cleaned up |
| Official metric executable with no arguments | Expected usage message; process exit 1 |
| ATR builder-style metadata through `imagefolder` | Failed as described above |
| ATR builder-style class labels through trainer preprocessing | Mismatch confirmed as described above |
| Existing ATR / Waymo environments: `pip check` | ATR passed; Waymo reported four conflicts |
| README and audit relative links; 9 README shell examples | All targets exist locally; examples pass `bash -n` |
| README media | 16 animated GIFs in a 4 × 4 gallery; two clips each; 256 × 192, 20 frames / 4 seconds, infinite loops; 7,422,625 bytes total |

No API calls for condition generation, full dataset conversion, model training, or checkpoint-based evaluation were run. Recorded video inspection and GIF encoding do not revalidate the underlying models or their reported metrics.
