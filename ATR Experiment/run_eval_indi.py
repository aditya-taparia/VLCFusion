#!/usr/bin/env python3
"""
Evaluate IR-only and VIS-only DETR models on the ATR dataset.

This reuses the same ATR data pipeline, transforms, and detection metrics as
`ensemble_trainer.py` / `run_eval_ensemble.py`, but instead of the fusion
multimodal model it runs the two single-modality checkpoints separately.

Usage (seen scenario — default IR & VIS checkpoints):
  python run_eval_indi.py  --model_dir_ir models/45k_seen_ir/checkpoint-24550  --model_dir_vis models/45k_seen_vis/checkpoint-24550  --scenario seen

Usage (unseen scenario):
  python run_eval_indi.py \
      --model_dir_ir models/45k_seen_ir/checkpoint-24550 \
      --model_dir_vis models/45k_seen_vis/checkpoint-24550 \
      --scenario unseen
"""

import argparse
import logging
import os
from typing import Any, Dict, Optional

import torch
from datasets import DatasetDict, load_dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments

from ensemble_trainer import (
    DataArguments,
    collate_fn_multimodal,
    compute_detection_metrics,
)


def prepare_eval_dataset(
    data_args: DataArguments,
    modality: str,
) -> DatasetDict:
    """
    Eval-only dataset prep for a single modality.
    This loads only the requested modality dataset so evaluation is completely
    decoupled from the other modality.
    """
    if modality not in {"ir", "vis"}:
        raise ValueError(f"Unsupported modality: {modality}")

    dataset_dir = data_args.ir_dataset_dir if modality == "ir" else data_args.visible_dataset_dir
    logging.info(f"Loading {modality.upper()} dataset from: {dataset_dir}")
    dataset = load_dataset("imagefolder", data_dir=dataset_dir)

    any_split = next(iter(dataset))
    if "image_id" not in dataset[any_split].column_names:
        def add_filename_id(example):
            return {"image_id": os.path.splitext(os.path.basename(example["image"].filename))[0]}
        dataset = dataset.map(add_filename_id)

    for split in dataset.keys():
        dataset[split] = dataset[split].sort("image_id")

    eval_transform = v2.Compose([
        v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def augment_and_transform_single_modality(examples: Dict[str, Any]) -> Dict[str, Any]:
        images_processed = []
        labels_out = []

        for image, objects, image_id in zip(
            examples["image"],
            examples["objects"],
            examples["image_id"],
        ):
            img_tv = tv_tensors.Image(image)

            raw_boxes = objects.get("bbox", [])
            if len(raw_boxes) > 0:
                boxes_tv = tv_tensors.BoundingBoxes(
                    raw_boxes,
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=(image.height, image.width),
                )
                labels_tv = torch.tensor(objects.get("category", []), dtype=torch.int64)
            else:
                boxes_tv = tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4)),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=(image.height, image.width),
                )
                labels_tv = torch.tensor([], dtype=torch.int64)

            output = eval_transform({
                "image": img_tv,
                "boxes": boxes_tv,
                "labels": labels_tv,
            })

            out_img = output["image"]
            out_boxes = output["boxes"]
            out_labels = output["labels"]

            images_processed.append(out_img)

            new_h, new_w = out_img.shape[-2:]
            if out_boxes.numel() > 0:
                x, y, w, h = out_boxes.unbind(-1)
                cx = (x + w / 2.0) / new_w
                cy = (y + h / 2.0) / new_h
                nw = w / new_w
                nh = h / new_h
                normalized_boxes = torch.stack([cx, cy, nw, nh], dim=-1)
            else:
                normalized_boxes = torch.zeros((0, 4), dtype=torch.float32)

            labels_out.append({
                "image_id": torch.tensor(int(image_id), dtype=torch.int64),
                "boxes": normalized_boxes,
                "class_labels": out_labels,
                "orig_size": torch.tensor([new_h, new_w], dtype=torch.int64),
            })

        return {
            "pixel_values": torch.stack(images_processed),
            "labels": labels_out,
        }

    for split in dataset.keys():
        dataset[split] = dataset[split].with_transform(
            augment_and_transform_single_modality
        )

    return dataset


def run_eval_indi(
    scenario: str = "seen",
    visible_dataset_dir: Optional[str] = None,
    ir_dataset_dir: Optional[str] = None,
    model_dir_ir: str = "models/45k_seen_ir/checkpoint-24550",
    model_dir_vis: str = "models/45k_seen_vis/checkpoint-24550",
    base_model_name: str = "facebook/detr-resnet-50",
    image_size: int = 480,
    batch_size: int = 8,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate IR-only and VIS-only DETR models on the ATR test set.

    Returns a dict with keys prefixed by `ir_` and `vis_` (e.g., `ir_map`, `vis_map`).
    """
    logger = logging.getLogger(__name__)

    if visible_dataset_dir is None:
        visible_dataset_dir = f"/data/ataparia/Darpa_datasets/atr_dataset/visible_{scenario}"
    if ir_dataset_dir is None:
        ir_dataset_dir = f"/data/ataparia/Darpa_datasets/atr_dataset/infrared_{scenario}"

    logger.info(f"Scenario: {scenario}")
    logger.info(f"Visible dataset: {visible_dataset_dir}")
    logger.info(f"IR dataset: {ir_dataset_dir}")
    logger.info(f"IR model weights: {model_dir_ir}")
    logger.info(f"VIS model weights: {model_dir_vis}")

    data_args = DataArguments(
        visible_dataset_dir=visible_dataset_dir,
        ir_dataset_dir=ir_dataset_dir,
        train_conditions_file="conditions/seen/vlm_train.json",
        val_conditions_file="conditions/seen/vlm_val.json",
        test_conditions_file=f"conditions/{scenario}/vlm_test.json",
        condition_indices_to_sample_str=None,
        image_size=image_size,
        num_classes=10,
    )

    id2label = {
        0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
        5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20",
    }
    label2id = {v: k for k, v in id2label.items()}

    model_config_ir = AutoConfig.from_pretrained(
        model_dir_ir,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label),
    )
    model_config_vis = AutoConfig.from_pretrained(
        model_dir_vis,
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

    logger.info("Preparing IR dataset...")
    ir_dataset = prepare_eval_dataset(data_args, modality="ir")
    logger.info("Preparing VIS dataset...")
    vis_dataset = prepare_eval_dataset(data_args, modality="vis")

    if "test" not in ir_dataset or "test" not in vis_dataset:
        raise ValueError("No 'test' split in dataset.")

    from transformers import AutoModelForObjectDetection

    logger.info("Loading IR-only model...")
    ir_model = AutoModelForObjectDetection.from_pretrained(model_dir_ir, config=model_config_ir)

    logger.info("Loading VIS-only model...")
    vis_model = AutoModelForObjectDetection.from_pretrained(model_dir_vis, config=model_config_vis)

    training_args = TrainingArguments(
        output_dir=os.path.join("indi_eval_output", f"{scenario}"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
    )

    def make_trainer(model):
        compute_metrics_fn = lambda eval_pred: compute_detection_metrics(
            eval_pred,
            image_processor=image_processor,
            id2label=id2label,
            threshold=threshold,
        )
        return Trainer(
            model=model,
            args=training_args,
            eval_dataset=None,
            tokenizer=image_processor,
            compute_metrics=compute_metrics_fn,
            data_collator=collate_fn_multimodal,
        )

    logger.info(f"Evaluating with confidence threshold={threshold}")
    logger.info("Running IR-only evaluation on test set...")
    trainer_ir = make_trainer(ir_model)
    metrics_ir = trainer_ir.evaluate(eval_dataset=ir_dataset["test"], metric_key_prefix="ir")

    logger.info("Running VIS-only evaluation on test set...")
    trainer_vis = make_trainer(vis_model)
    metrics_vis = trainer_vis.evaluate(eval_dataset=vis_dataset["test"], metric_key_prefix="vis")

    # Merge metrics dicts (they already have distinct prefixes)
    merged: Dict[str, float] = {}
    merged.update(metrics_ir)
    merged.update(metrics_vis)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Evaluate individual IR/VIS DETR models on ATR test set.")
    parser.add_argument("--scenario", type=str, default="seen", choices=["seen", "unseen"],
                        help="Evaluation scenario: 'seen' or 'unseen' (controls dataset dirs and test conditions).")
    parser.add_argument("--visible_dataset_dir", type=str, default=None,
                        help="Override visible dataset dir (default: auto from --scenario).")
    parser.add_argument("--ir_dataset_dir", type=str, default=None,
                        help="Override IR dataset dir (default: auto from --scenario).")
    parser.add_argument("--model_dir_ir", type=str, default="models/45k_seen_ir/checkpoint-24550",
                        help="Path to IR DETR checkpoint.")
    parser.add_argument("--model_dir_vis", type=str, default="models/45k_seen_vis/checkpoint-24550",
                        help="Path to Visible DETR checkpoint.")
    parser.add_argument("--base_model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Detection confidence threshold for mAP.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    metrics = run_eval_indi(
        scenario=args.scenario,
        visible_dataset_dir=args.visible_dataset_dir,
        ir_dataset_dir=args.ir_dataset_dir,
        model_dir_ir=args.model_dir_ir,
        model_dir_vis=args.model_dir_vis,
        base_model_name=args.base_model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    logger.info("===== Individual Model Evaluation Results =====")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k}: {v}")
    print(metrics)
    return metrics


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    main()

