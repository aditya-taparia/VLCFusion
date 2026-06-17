#!/usr/bin/env python3
"""
Standalone evaluation script for the multimodal ensemble (e.g. VLCFusion).

This uses the SAME torchvision-v2 preprocessing pipeline that the trainer uses
for its validation/test evaluation (anisotropic Resize to image_size x image_size
+ ImageNet normalization, with bounding boxes carried through as tv_tensors).

This matters: an earlier version of this script used an albumentations + image
processor pipeline (aspect-preserving resize + padding). That geometry does not
match how the models are trained (square resize), so for ATR's very small
targets the predicted boxes were shifted by tens of pixels and mAP collapsed to
~0. Reusing the trainer's exact functions guarantees the standalone numbers
match the trainer's own test_results.json.

Usage (seen scenario - default):
  python run_eval_ensemble.py --checkpoint_dir models/45k_vlcfusion_2blocks_7cond/checkpoint-XXXX --ensemble_method VLCFusion

Usage (unseen scenario):
  python run_eval_ensemble.py --checkpoint_dir models/45k_vlcfusion_2blocks_7cond/checkpoint-XXXX --ensemble_method VLCFusion --scenario unseen
"""

import argparse
import logging
import os
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torchvision.transforms import v2
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments

# Reuse the trainer's exact preprocessing / metrics so results are identical to
# the trainer's own validation/test evaluation.
from ensemble_trainer import (
    DataArguments,
    augment_and_transform_batch_multimodal,
    collate_fn_multimodal,
    compute_detection_metrics,
    get_effective_conditions_and_count,
)
from multimodal_detr import MultimodalDetr


# ---------------------------------------------------------------------------
# Dataset preparation (torch-v2 pipeline, identical to trainer's eval path)
# ---------------------------------------------------------------------------

def prepare_eval_dataset(
    data_args: DataArguments,
    image_processor: AutoImageProcessor,
    test_conditions: Dict[str, Any],
    indices_to_sample_conditions: Optional[np.ndarray],
) -> DatasetDict:
    """Eval-only dataset prep using the trainer's torch-v2 transform on the test split."""
    logging.info(f"Loading visible dataset from: {data_args.visible_dataset_dir}")
    vis_data = load_dataset("imagefolder", data_dir=data_args.visible_dataset_dir)

    logging.info(f"Loading IR dataset from: {data_args.ir_dataset_dir}")
    ir_data = load_dataset("imagefolder", data_dir=data_args.ir_dataset_dir)

    def prefix_and_align(dataset, prefix):
        any_split = next(iter(dataset))
        if "image_id" not in dataset[any_split].column_names:
            def add_filename_id(example):
                return {"image_id": os.path.splitext(os.path.basename(example["image"].filename))[0]}
            dataset = dataset.map(add_filename_id)
        for split in dataset.keys():
            dataset[split] = dataset[split].sort("image_id").shuffle(seed=42)
            dataset[split] = dataset[split].rename_columns(
                {col: f"{prefix}_{col}" for col in dataset[split].column_names}
            )
        return dataset

    vis_data = prefix_and_align(vis_data, "vis")
    ir_data = prefix_and_align(ir_data, "ir")

    if "test" not in vis_data or "test" not in ir_data:
        raise ValueError("No 'test' split found in the visible/IR datasets.")

    if len(vis_data["test"]) != len(ir_data["test"]):
        logging.warning(
            f"Length mismatch in test: Vis {len(vis_data['test'])} vs IR {len(ir_data['test'])}"
        )

    combined_test = concatenate_datasets([vis_data["test"], ir_data["test"]], axis=1)
    combined_dataset = DatasetDict({"test": combined_test})

    # Trainer's eval transform: square resize + ImageNet normalize (boxes carried along).
    eval_torch_transform = v2.Compose([
        v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    combined_dataset["test"] = combined_dataset["test"].with_transform(
        partial(
            augment_and_transform_batch_multimodal,
            transform=eval_torch_transform,
            image_processor=image_processor,
            all_conditions_data={"test": test_conditions},
            current_split_name="test",
            indices_to_sample_conditions=indices_to_sample_conditions,
        )
    )

    return combined_dataset


def run_eval(
    checkpoint_dir: str,
    ensemble_method: str = "VLCFusion",
    scenario: str = "seen",
    visible_dataset_dir: str = None,
    ir_dataset_dir: str = None,
    condition_indices: str = "16,13,1,11,15,19,18",
    model_dir_ir: str = "models/45k_seen_ir/checkpoint-24550",
    model_dir_vis: str = "models/45k_seen_vis/checkpoint-24550",
    base_model_name: str = "facebook/detr-resnet-50",
    image_size: int = 480,
    batch_size: int = 8,
    threshold: float = 0.0,
    num_vlc_blocks: int = 2,
):
    """Run evaluation with the trainer's torch-v2 preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if visible_dataset_dir is None:
        visible_dataset_dir = f"/data/ataparia/Darpa_datasets/atr_dataset/visible_{scenario}"
    if ir_dataset_dir is None:
        ir_dataset_dir = f"/data/ataparia/Darpa_datasets/atr_dataset/infrared_{scenario}"

    logger.info(f"Scenario: {scenario}")
    logger.info(f"Visible dataset: {visible_dataset_dir}")
    logger.info(f"IR dataset: {ir_dataset_dir}")

    data_args = DataArguments(
        visible_dataset_dir=visible_dataset_dir,
        ir_dataset_dir=ir_dataset_dir,
        train_conditions_file="conditions/seen/vlm_train.json",
        val_conditions_file="conditions/seen/vlm_val.json",
        test_conditions_file=f"conditions/{scenario}/vlm_test.json",
        condition_indices_to_sample_str=condition_indices,
        image_size=image_size,
        num_classes=10,
    )

    indices_to_sample_arr, n_conditions_eff, train_cond, val_cond, test_cond = get_effective_conditions_and_count(
        data_args.train_conditions_file,
        data_args.val_conditions_file,
        data_args.test_conditions_file,
        data_args.condition_indices_to_sample_str,
    )

    id2label = {
        0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
        5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20",
    }
    label2id = {v: k for k, v in id2label.items()}

    model_config = AutoConfig.from_pretrained(
        base_model_name,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label),
    )

    image_processor = AutoImageProcessor.from_pretrained(
        base_model_name,
        do_resize=True,
        size={"max_height": data_args.image_size, "max_width": data_args.image_size},
        do_pad=True,
        pad_size={"height": data_args.image_size, "width": data_args.image_size},
    )

    logger.info("Preparing datasets...")
    combined_dataset = prepare_eval_dataset(
        data_args, image_processor, test_cond, indices_to_sample_arr,
    )

    if "test" not in combined_dataset:
        raise ValueError("No 'test' split in dataset.")

    model = MultimodalDetr(
        model_name_1=model_dir_ir,
        model_name_2=model_dir_vis,
        config=model_config,
        ensemble_method=ensemble_method,
        n_conditions=n_conditions_eff,
        num_vlc_blocks=num_vlc_blocks,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "eval_output"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
    )

    eval_compute_metrics_fn = partial(
        compute_detection_metrics,
        image_processor=image_processor,
        id2label=id2label,
        threshold=threshold,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset["test"],
        eval_dataset=combined_dataset["test"],
        tokenizer=image_processor,
        compute_metrics=eval_compute_metrics_fn,
        data_collator=collate_fn_multimodal,
    )

    logger.info(f"Loading checkpoint: {checkpoint_dir}")
    # Manual load with assign=True. MultimodalDetr is a custom nn.Module and does
    # not define _keys_to_ignore_on_save, which trips up Trainer._load_from_checkpoint's
    # post-load warning path. assign=True also avoids the "copying to a meta parameter
    # (no-op)" issue that silently left backbone buffers unloaded.
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file as _safe_load
        state_dict = _safe_load(st_path)
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint_dir}.")

    # Backward-compat: checkpoints trained before the VLC-block refactor stored the
    # two stacked blocks as named attributes (.vlc_fused. / .vlc_fused2.). The
    # refactored model stores them as a ModuleList (.vlc_blocks.0. / .vlc_blocks.1.).
    # Remap old keys so legacy 2-block checkpoints (e.g. the paper's reference) load.
    if any(".vlc_fused." in k or ".vlc_fused2." in k for k in state_dict):
        remapped = {}
        for k, v in state_dict.items():
            if ".vlc_fused2." in k:
                k = k.replace(".vlc_fused2.", ".vlc_blocks.1.")
            elif ".vlc_fused." in k:
                k = k.replace(".vlc_fused.", ".vlc_blocks.0.")
            remapped[k] = v
        state_dict = remapped
        logger.info("Remapped legacy vlc_fused/vlc_fused2 keys -> vlc_blocks.0/.1")

    load_result = trainer.model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(
        f"Loaded checkpoint: {len(load_result.missing_keys)} missing keys, "
        f"{len(load_result.unexpected_keys)} unexpected keys."
    )
    if load_result.missing_keys:
        logger.warning(f"Missing keys (first 10): {load_result.missing_keys[:10]}")

    logger.info("Running evaluation on test set...")
    metrics = trainer.evaluate(eval_dataset=combined_dataset["test"], metric_key_prefix="test")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal ensemble on test set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint (e.g. models/45k_vlcfusion_2blocks_7cond/checkpoint-XXXX)")
    parser.add_argument("--ensemble_method", type=str, default="VLCFusion",
                        help="Ensemble method (must match how the checkpoint was trained)")
    parser.add_argument("--scenario", type=str, default="seen", choices=["seen", "unseen"],
                        help="Evaluation scenario: 'seen' or 'unseen' (switches dataset dirs and test conditions)")
    parser.add_argument("--visible_dataset_dir", type=str, default=None,
                        help="Override visible dataset dir (default: auto from --scenario)")
    parser.add_argument("--ir_dataset_dir", type=str, default=None,
                        help="Override IR dataset dir (default: auto from --scenario)")
    parser.add_argument("--condition_indices", type=str, default="16,13,1,11,15,19,18",
                        help="Comma-separated 1-based condition indices (must match training)")
    parser.add_argument("--model_dir_ir", type=str, default="models/45k_seen_ir/checkpoint-24550")
    parser.add_argument("--model_dir_vis", type=str, default="models/45k_seen_vis/checkpoint-24550")
    parser.add_argument("--base_model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.0, help="Detection confidence threshold for mAP")
    parser.add_argument("--num_vlc_blocks", type=int, default=2, help="Number of VLC blocks (must match training; VLCFusion only)")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    metrics = run_eval(
        checkpoint_dir=args.checkpoint_dir,
        ensemble_method=args.ensemble_method,
        scenario=args.scenario,
        visible_dataset_dir=args.visible_dataset_dir,
        ir_dataset_dir=args.ir_dataset_dir,
        condition_indices=args.condition_indices,
        model_dir_ir=args.model_dir_ir,
        model_dir_vis=args.model_dir_vis,
        base_model_name=args.base_model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_vlc_blocks=args.num_vlc_blocks,
    )
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
