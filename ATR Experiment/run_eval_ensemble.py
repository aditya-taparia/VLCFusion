#!/usr/bin/env python3
"""
Standalone evaluation script for multimodal ensemble (e.g. VLCFusion).
Uses the original albumentations + image_processor data pipeline so that
results are consistent with models trained under that preprocessing.

Usage (seen scenario — default):
  python run_eval_ensemble.py --checkpoint_dir models/45k_corssbam_fusion_7_cond_fusion/checkpoint-4275 --ensemble_method VLCFusion

Usage (unseen scenario):
  python run_eval_ensemble.py --checkpoint_dir models/45k_corssbam_fusion_7_cond_fusion/checkpoint-4275 --ensemble_method VLCFusion --scenario unseen
"""

import argparse
import logging
import os
import sys
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoConfig, AutoImageProcessor, Trainer, TrainingArguments
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ensemble_trainer import (
    DataArguments,
    MultimodalTrainer,
    get_effective_conditions_and_count,
)
from multimodal_detr import MultimodalDetr


# ---------------------------------------------------------------------------
# Data helpers (original albumentations + image_processor pipeline)
# ---------------------------------------------------------------------------

def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float, ...]]
) -> Dict[str, Any]:
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        annotations.append({
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        })
    return {"image_id": image_id, "annotations": annotations}


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]], device=boxes.device)
    return boxes


class ModelOutput:
    def __init__(self, logits: torch.Tensor, pred_boxes: torch.Tensor):
        self.logits = logits
        self.pred_boxes = pred_boxes


def augment_and_transform_batch_legacy(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    conditions_data: Dict[str, Any],
    indices_to_sample: Optional[np.ndarray],
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Original preprocessing: albumentations + image_processor for each modality."""

    ir_images, ir_annotations = [], []
    for image_id, image, objects in zip(
        examples["ir_image_id"], examples["ir_image"], examples["ir_objects"]
    ):
        image_np = np.array(image.convert("RGB"))
        output = transform(image=image_np, bboxes=objects["bbox"], category=objects["category"])
        ir_images.append(output["image"])
        ir_annotations.append(format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        ))

    ir_result = image_processor(images=ir_images, annotations=ir_annotations, return_tensors="pt")

    vis_images = []
    for image in examples["vis_image"]:
        image_np = np.array(image.convert("RGB"))
        output = transform(image=image_np, bboxes=[], category=[])
        vis_images.append(output["image"])

    vis_result = image_processor(images=vis_images, annotations=[{"image_id": 0, "annotations": []}] * len(vis_images), return_tensors="pt")

    ir_result["pixel_values"] = torch.cat([ir_result["pixel_values"], vis_result["pixel_values"]], dim=1)

    for label in ir_result["labels"]:
        img_id_val = label["image_id"]
        image_id_str = str(img_id_val.item()) if hasattr(img_id_val, "item") else str(img_id_val)
        conditions = conditions_data.get(image_id_str)

        if conditions is None:
            n = indices_to_sample.shape[0] if indices_to_sample is not None else 1
            cond_tensor = torch.zeros(n, dtype=torch.float)
        else:
            if isinstance(conditions[0], bool):
                cond_tensor = torch.tensor([1.0 if c else 0.0 for c in conditions], dtype=torch.float)
            else:
                cond_tensor = torch.tensor(conditions, dtype=torch.float)
            if indices_to_sample is not None:
                cond_tensor = cond_tensor[indices_to_sample - 1]

        label["conditions"] = cond_tensor

    if not return_pixel_mask:
        ir_result.pop("pixel_mask", None)

    return ir_result


def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": [x["labels"] for x in batch]}
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics_legacy(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    for batch in targets:
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    if not post_processed_targets or not post_processed_predictions:
        return {"map": 0.0}

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    return {k: round(v.item(), 4) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_eval_dataset(
    data_args: DataArguments,
    image_processor: AutoImageProcessor,
    conditions_data: Dict[str, Any],
    indices_to_sample_conditions: Optional[np.ndarray],
) -> DatasetDict:
    """Eval-only dataset prep using original albumentations + image_processor pipeline."""
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

    eval_alb_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    for split in combined_dataset.keys():
        combined_dataset[split] = combined_dataset[split].with_transform(
            partial(
                augment_and_transform_batch_legacy,
                transform=eval_alb_transform,
                image_processor=image_processor,
                conditions_data=conditions_data,
                indices_to_sample=indices_to_sample_conditions,
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
):
    """Run evaluation with the original albumentations + image_processor pipeline."""
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
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "eval_output"),
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
    )

    eval_compute_metrics_fn = partial(
        compute_metrics_legacy,
        image_processor=image_processor,
        threshold=threshold,
        id2label=id2label,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset["test"],
        eval_dataset=combined_dataset["test"],
        tokenizer=image_processor,
        compute_metrics=eval_compute_metrics_fn,
        data_collator=collate_fn,
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
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
