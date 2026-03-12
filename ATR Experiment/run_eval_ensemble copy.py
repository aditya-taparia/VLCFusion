#!/usr/bin/env python3
"""
Standalone evaluation script for multimodal ensemble (e.g. CrossCBAM_AdaLN).
Uses the exact same data pipeline, collate, and metrics as ensemble_trainer.py
so that test results match training-time evaluation.

Usage (seen scenario — default):
  python run_eval_ensemble.py --checkpoint_dir models/45k_corssbam_fusion_7_cond_fusion/checkpoint-4275 --ensemble_method CrossCBAM_AdaLN

Usage (unseen scenario):
  python run_eval_ensemble.py --checkpoint_dir models/45k_corssbam_fusion_7_cond_fusion/checkpoint-4275 --ensemble_method CrossCBAM_AdaLN --scenario unseen
"""

import argparse
import logging
import os
import sys
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torchvision.transforms import v2
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments

from ensemble_trainer import (
    DataArguments,
    MultimodalTrainer,
    augment_and_transform_batch_multimodal,
    collate_fn_multimodal,
    compute_detection_metrics,
    get_effective_conditions_and_count,
)
from multimodal_detr import MultimodalDetr


def prepare_eval_dataset(
    data_args: DataArguments,
    image_processor: AutoImageProcessor,
    all_conditions: Dict[str, Dict[str, Any]],
    indices_to_sample_conditions: Optional[np.ndarray],
) -> DatasetDict:
    """Eval-only dataset prep. Loads only available splits (no train required)."""
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

    processed_splits = {}
    for split in vis_data.keys():
        if split not in ir_data:
            continue
        if len(vis_data[split]) != len(ir_data[split]):
            logging.warning(f"Length mismatch in {split}: Vis {len(vis_data[split])} vs IR {len(ir_data[split])}")
        processed_splits[split] = concatenate_datasets([vis_data[split], ir_data[split]], axis=1)

    combined_dataset = DatasetDict(processed_splits)

    eval_transform = v2.Compose([
        v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_kwargs = {
        "image_processor": image_processor,
        "all_conditions_data": all_conditions,
        "indices_to_sample_conditions": indices_to_sample_conditions,
    }

    for split in combined_dataset.keys():
        combined_dataset[split] = combined_dataset[split].with_transform(
            partial(augment_and_transform_batch_multimodal,
                    transform=eval_transform, current_split_name=split, **transform_kwargs)
        )

    return combined_dataset


def run_eval(
    checkpoint_dir: str,
    ensemble_method: str = "CrossCBAM_AdaLN",
    scenario: str = "seen",
    visible_dataset_dir: str = None,
    ir_dataset_dir: str = None,
    condition_indices: str = "16,13,1,11,15,19,18",
    model_dir_ir: str = "models/45k_seen_ir/checkpoint-24550",
    model_dir_vis: str = "models/45k_seen_vis/checkpoint-24550",
    base_model_name: str = "facebook/detr-resnet-50",
    image_size: int = 480,
    batch_size: int = 8,
    threshold: float = 0.5,
):
    """Run evaluation with the same pipeline as ensemble_trainer. Returns test metrics dict."""
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
    all_conditions = {"train": train_cond, "validation": val_cond, "test": test_cond}

    id2label = {
        0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
        5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20",
    }
    label2id = {v: k for k, v in id2label.items()}

    model_config = AutoConfig.from_pretrained(
        model_dir_ir,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label),
    )

    image_processor = AutoImageProcessor.from_pretrained(
        base_model_name,
        do_resize=True,
        size={"longest_edge": data_args.image_size, "shortest_edge": data_args.image_size},
        do_pad=True,
    )

    logger.info("Preparing datasets...")
    combined_dataset = prepare_eval_dataset(data_args, image_processor, all_conditions, indices_to_sample_arr)

    if "test" not in combined_dataset:
        raise ValueError("No 'test' split in dataset.")

    model = MultimodalDetr(
        model_name_1=model_dir_ir,
        model_name_2=model_dir_vis,
        config=model_config,
        ensemble_method=ensemble_method,
        n_conditions=n_conditions_eff,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "eval_output"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
    )

    compute_metrics_fn = lambda eval_pred: compute_detection_metrics(
        eval_pred,
        image_processor=image_processor,
        id2label=id2label,
        threshold=threshold,
    )

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        eval_dataset=combined_dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics_fn,
        data_collator=collate_fn_multimodal,
    )

    logger.info(f"Loading checkpoint: {checkpoint_dir}")
    try:
        trainer._load_from_checkpoint(checkpoint_dir)
    except Exception as e:
        err_msg = str(e).lower()
        if "safetensor" in type(e).__name__.lower() or "header too small" in err_msg:
            bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.isfile(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                trainer.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded from {bin_path} (safetensors corrupted).")
            else:
                raise FileNotFoundError(f"No pytorch_model.bin at {bin_path}.") from e
        else:
            raise

    logger.info("Running evaluation on test set...")
    metrics = trainer.evaluate(eval_dataset=combined_dataset["test"], metric_key_prefix="test")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal ensemble on test set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint (e.g. models/45k_corssbam_fusion_7_cond_fusion/checkpoint-4275)")
    parser.add_argument("--ensemble_method", type=str, default="CrossCBAM_AdaLN",
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
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold for mAP")
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
    )
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    main()
