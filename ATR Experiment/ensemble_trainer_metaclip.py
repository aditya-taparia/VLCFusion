"""
Training script for MultimodalDetrMetaCLIP.

Instead of loading precalculated condition JSONs, this trainer uses the
visible (RGB) image itself: each batch includes an additional
`clip_pixel_values` tensor that is produced by the MetaCLIP processor and
fed into the frozen MetaCLIP vision encoder inside the model to produce
conditioning vectors on-the-fly.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from torchvision.transforms import v2
from torchvision import tv_tensors
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint

from multimodal_detr_metaclip import MultimodalDetrMetaCLIP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_dir_1: str = field(
        default="models/45k_seen_ir/checkpoint-24550",
        metadata={"help": "Path to the IR DETR checkpoint."},
    )
    model_dir_2: str = field(
        default="models/45k_seen_vis/checkpoint-24550",
        metadata={"help": "Path to the Visible DETR checkpoint."},
    )
    base_model_name: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Base DETR model name for image processor."},
    )
    clip_model_name: str = field(
        default="facebook/metaclip-2-worldwide-s16-384",
        metadata={"help": "MetaCLIP-2 model for live conditioning. "
                  "Options: facebook/metaclip-2-worldwide-s16-384 (fastest), "
                  "metaclip-2-worldwide-l14 (default), "
                  "metaclip-2-worldwide-huge-quickgelu (best quality)."},
    )


@dataclass
class DataArguments:
    visible_dataset_dir: str = field(
        default="/data/ataparia/Darpa_datasets/atr_dataset/visible_seen",
        metadata={"help": "Path to the visible spectrum dataset."},
    )
    ir_dataset_dir: str = field(
        default="/data/ataparia/Darpa_datasets/atr_dataset/infrared_seen",
        metadata={"help": "Path to the infrared spectrum dataset."},
    )
    image_size: int = field(default=480, metadata={"help": "Target image size."})
    num_classes: int = field(default=10, metadata={"help": "Number of object classes."})


@dataclass
class FusionTrainingArguments:
    fusion_lr_multiplier: float = field(
        default=1.0,
        metadata={"help": "LR multiplier for fusion layers."},
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float, ...]]
) -> Dict[str, Any]:
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        annotations.append({
            "image_id": image_id, "category_id": category,
            "iscrowd": 0, "area": area, "bbox": list(bbox),
        })
    return {"image_id": image_id, "annotations": annotations}


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    boxes_converted = center_to_corners_format(boxes)
    height, width = image_size
    boxes_converted = boxes_converted * torch.tensor(
        [[width, height, width, height]], device=boxes.device
    )
    return boxes_converted


# ---------------------------------------------------------------------------
# Data transforms — no conditions, but we add clip_pixel_values
# ---------------------------------------------------------------------------

def augment_and_transform_batch_multimodal(
    examples: Mapping[str, Any],
    transform: v2.Compose,
    image_processor: AutoImageProcessor,
    clip_processor: AutoProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations to IR and Visible, and also produce MetaCLIP inputs
    from the visible image."""

    ir_images_processed = []
    vis_images_processed = []
    combined_labels = []

    vis_pil_images = list(examples["vis_image"])

    # Batch MetaCLIP preprocessing — one call for all visible images
    clip_inputs = clip_processor(images=vis_pil_images, return_tensors="pt")
    clip_pixel_values_batch = clip_inputs["pixel_values"]  # [B, 3, 224, 224]

    for ir_img, vis_img, ir_objs, ir_id in zip(
        examples["ir_image"],
        vis_pil_images,
        examples["ir_objects"],
        examples["ir_image_id"],
    ):
        if ir_img.size != vis_img.size:
            ir_img = ir_img.resize(vis_img.size)

        # --- TVTensor wrapping ---
        img_ir_tv = tv_tensors.Image(ir_img)
        img_vis_tv = tv_tensors.Image(vis_img)

        raw_boxes = ir_objs.get("bbox", [])
        if len(raw_boxes) > 0:
            boxes_tv = tv_tensors.BoundingBoxes(
                raw_boxes,
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=(ir_img.height, ir_img.width),
            )
            labels_tv = torch.tensor(ir_objs.get("category", []), dtype=torch.int64)
        else:
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)),
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=(ir_img.height, ir_img.width),
            )
            labels_tv = torch.tensor([], dtype=torch.int64)

        output = transform({
            "ir": img_ir_tv, "vis": img_vis_tv,
            "boxes": boxes_tv, "labels": labels_tv,
        })

        out_ir = output["ir"]
        out_vis = output["vis"]
        out_boxes = output["boxes"]
        out_labels = output["labels"]

        ir_images_processed.append(out_ir)
        vis_images_processed.append(out_vis)

        new_h, new_w = out_ir.shape[-2:]

        if out_boxes.numel() > 0:
            x, y, w, h = out_boxes.unbind(-1)
            cx = (x + w / 2.0) / new_w
            cy = (y + h / 2.0) / new_h
            nw = w / new_w
            nh = h / new_h
            normalized_boxes = torch.stack([cx, cy, nw, nh], dim=-1)
        else:
            normalized_boxes = torch.zeros((0, 4), dtype=torch.float32)

        id_tensor = torch.tensor(int(ir_id), dtype=torch.int64)
        size_tensor = torch.tensor([new_h, new_w], dtype=torch.int64)

        formatted_ann = {
            "image_id": id_tensor,
            "boxes": normalized_boxes,
            "class_labels": out_labels,
            "orig_size": size_tensor,
        }
        combined_labels.append(formatted_ann)

    pixel_values = torch.cat([
        torch.stack(ir_images_processed),
        torch.stack(vis_images_processed),
    ], dim=1)

    final_batch = {
        "pixel_values": pixel_values,
        "clip_pixel_values": clip_pixel_values_batch,
        "labels": combined_labels,
    }

    if not return_pixel_mask:
        final_batch.pop("pixel_mask", None)

    return final_batch


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn_multimodal(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    clip_pixel_values = torch.stack([x["clip_pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch]

    collated = {
        "pixel_values": pixel_values,
        "clip_pixel_values": clip_pixel_values,
        "labels": labels,
    }
    if "pixel_mask" in batch[0] and batch[0]["pixel_mask"] is not None:
        collated["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return collated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_detection_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    id2label: Mapping[int, str],
    threshold: float = 0.0,
) -> Mapping[str, float]:
    predictions_batches = evaluation_results.predictions
    targets_batches = evaluation_results.label_ids

    processed_predictions, processed_targets = [], []

    for batch_pred, batch_targets in zip(predictions_batches, targets_batches):
        if isinstance(batch_pred, tuple):
            b_logits = torch.tensor(batch_pred[1])
            b_boxes = torch.tensor(batch_pred[2])
        elif isinstance(batch_pred, dict):
            b_logits = torch.tensor(batch_pred["logits"])
            b_boxes = torch.tensor(batch_pred["pred_boxes"])
        else:
            b_logits = torch.tensor(batch_pred.logits)
            b_boxes = torch.tensor(batch_pred.pred_boxes)

        for i, target in enumerate(batch_targets):
            if "orig_size" in target:
                sz = target["orig_size"]
                img_h, img_w = sz.tolist() if hasattr(sz, "tolist") else sz
            else:
                img_h, img_w = 480, 480

            t_boxes = torch.tensor(target["boxes"])
            t_boxes = convert_bbox_yolo_to_pascal(t_boxes, (img_h, img_w))
            t_labels = torch.tensor(target["class_labels"])
            processed_targets.append({"boxes": t_boxes, "labels": t_labels})

            p_logits = b_logits[i]
            p_boxes = b_boxes[i]
            target_sizes = torch.tensor([[img_h, img_w]])
            output = ModelOutput(logits=p_logits.unsqueeze(0), pred_boxes=p_boxes.unsqueeze(0))

            post_processed = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes,
            )
            processed_predictions.extend(post_processed)

    if not processed_targets or not processed_predictions:
        logging.warning("No valid targets or predictions to compute metrics.")
        return {"map": 0.0}

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(processed_predictions, processed_targets)
    metrics = metric.compute()

    final_metrics = {}
    for k, v in metrics.items():
        if k not in ("classes", "map_per_class", "mar_100_per_class"):
            final_metrics[k] = round(v.item(), 4)

    if "map_per_class" in metrics:
        for class_id, class_map in zip(metrics["classes"], metrics["map_per_class"]):
            class_name = id2label.get(class_id.item(), str(class_id.item()))
            final_metrics[f"map_{class_name}"] = round(class_map.item(), 4)

    return final_metrics


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_datasets(
    data_args: DataArguments,
    image_processor: AutoImageProcessor,
    clip_processor: AutoProcessor,
) -> DatasetDict:
    logging.info(f"Loading visible dataset from: {data_args.visible_dataset_dir}")
    vis_custom_data = load_dataset("imagefolder", data_dir=data_args.visible_dataset_dir)

    logging.info(f"Loading IR dataset from: {data_args.ir_dataset_dir}")
    ir_custom_data = load_dataset("imagefolder", data_dir=data_args.ir_dataset_dir)

    def ensure_alignment_columns(dataset, prefix):
        if "image_id" not in dataset["train"].column_names:
            def add_filename_id(example):
                return {"image_id": os.path.splitext(os.path.basename(example["image"].filename))[0]}
            dataset = dataset.map(add_filename_id)

        for split in dataset.keys():
            dataset[split] = dataset[split].sort("image_id").shuffle(seed=42)
            dataset[split] = dataset[split].rename_columns(
                {col: f"{prefix}_{col}" for col in dataset[split].column_names}
            )
        return dataset

    vis_custom_data = ensure_alignment_columns(vis_custom_data, "vis")
    ir_custom_data = ensure_alignment_columns(ir_custom_data, "ir")

    processed_splits = {}
    for split in ["train", "validation", "test"]:
        if split not in vis_custom_data or split not in ir_custom_data:
            continue
        if len(vis_custom_data[split]) != len(ir_custom_data[split]):
            logging.warning(
                f"Length mismatch in {split}: Vis {len(vis_custom_data[split])} vs IR {len(ir_custom_data[split])}"
            )
        combined_split = concatenate_datasets([vis_custom_data[split], ir_custom_data[split]], axis=1)
        processed_splits[split] = combined_split

    combined_dataset = DatasetDict(processed_splits)

    # --- Transforms ---
    train_torch_transform = v2.Compose([
        v2.RandomChoice([
            v2.Compose([
                v2.Resize(size=data_args.image_size, antialias=True),
                v2.RandomCrop(size=(data_args.image_size, data_args.image_size), pad_if_needed=True),
            ]),
            v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        ], p=[0.2, 0.8]),
        v2.RandomApply([
            v2.RandomChoice([
                v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                v2.GaussianBlur(kernel_size=7, sigma=(2.0, 5.0)),
            ])
        ], p=0.1),
        v2.RandomPerspective(distortion_scale=0.1, p=0.1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        v2.RandomApply([v2.ColorJitter(hue=0.1, saturation=0.1)], p=0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(),
    ])

    eval_torch_transform = v2.Compose([
        v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_kwargs = {
        "image_processor": image_processor,
        "clip_processor": clip_processor,
    }

    combined_dataset["train"] = combined_dataset["train"].with_transform(
        partial(augment_and_transform_batch_multimodal,
                transform=train_torch_transform, **transform_kwargs)
    )
    if "validation" in combined_dataset:
        combined_dataset["validation"] = combined_dataset["validation"].with_transform(
            partial(augment_and_transform_batch_multimodal,
                    transform=eval_torch_transform, **transform_kwargs)
        )
    if "test" in combined_dataset:
        combined_dataset["test"] = combined_dataset["test"].with_transform(
            partial(augment_and_transform_batch_multimodal,
                    transform=eval_torch_transform, **transform_kwargs)
        )

    return combined_dataset


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class MultimodalMetaCLIPTrainer(Trainer):
    """Trainer that creates separate optimizer groups for fusion layers and
    excludes all frozen parameters (backbones + MetaCLIP conditioner)."""

    def __init__(self, fusion_args: Optional[FusionTrainingArguments] = None, **kwargs):
        self.fusion_args = fusion_args or FusionTrainingArguments()
        super().__init__(**kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        fa = self.fusion_args
        if fa.fusion_lr_multiplier == 1.0:
            return super().create_optimizer()

        model = self.model
        base_lr = self.args.learning_rate

        frozen_prefixes = (
            "backbone_ir", "input_projection_ir",
            "backbone_rgb", "input_projection_rgb",
            "conditioner",
        )
        fusion_prefixes = ("transform_layer", "transform_queries")

        groups: Dict[str, list] = {
            "fusion_decay": [], "fusion_no_decay": [],
            "other_decay": [], "other_no_decay": [],
        }

        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                continue
            is_fusion = any(name.startswith(p) for p in fusion_prefixes)
            if is_fusion:
                prefix = "fusion"
            else:
                if not param.requires_grad:
                    continue
                prefix = "other"
            no_wd = name.endswith(".bias") or ".norm" in name
            groups[f"{prefix}_{'no_decay' if no_wd else 'decay'}"].append(param)

        wd = self.args.weight_decay
        fusion_lr = base_lr * fa.fusion_lr_multiplier

        optimizer_grouped = []
        if groups["fusion_decay"]:
            optimizer_grouped.append({"params": groups["fusion_decay"], "lr": fusion_lr, "weight_decay": wd})
        if groups["fusion_no_decay"]:
            optimizer_grouped.append({"params": groups["fusion_no_decay"], "lr": fusion_lr, "weight_decay": 0.0})
        if groups["other_decay"]:
            optimizer_grouped.append({"params": groups["other_decay"], "lr": base_lr, "weight_decay": wd})
        if groups["other_no_decay"]:
            optimizer_grouped.append({"params": groups["other_no_decay"], "lr": base_lr, "weight_decay": 0.0})

        self.optimizer = torch.optim.AdamW(optimizer_grouped)
        return self.optimizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_args = [
        "--output_dir", "models/45k_crosscbam_dit_v2_metaclip",
        "--run_name", "45k_crosscbam_dit_v2_metaclip",

        "--num_train_epochs", "100",
        "--per_device_train_batch_size", "64",
        "--gradient_accumulation_steps", "16",
        "--learning_rate", "5e-5",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        "--bf16", "True",

        "--metric_for_best_model", "eval_map",
        "--greater_is_better", "True",
        "--load_best_model_at_end", "True",
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--save_total_limit", "5",
        "--logging_steps", "100",
        "--report_to", "wandb",

        "--remove_unused_columns", "False",
        "--eval_do_concat_batches", "False",
        "--do_train",
        "--do_eval",

        "--dataloader_num_workers", "16",
        "--dataloader_pin_memory", "True",
    ]

    combined_args = default_args + sys.argv[1:]

    parser = HfArgumentParser((ModelArguments, DataArguments, FusionTrainingArguments, TrainingArguments))
    model_args, data_args, fusion_args, training_args = parser.parse_args_into_dataclasses(args=combined_args)

    # --- Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Fusion training parameters {fusion_args}")

    # --- Checkpoint ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # --- Labels ---
    _categories_to_tgttype = {
        0: "PICKUP", 1: "SUV", 2: "BTR70", 3: "BRDM2", 4: "BMP2",
        5: "T72", 6: "ZSU23", 7: "2S3", 8: "MTLB", 9: "D20",
    }
    id2label = {k: v for k, v in _categories_to_tgttype.items() if k < data_args.num_classes}
    label2id = {v: k for k, v in id2label.items()}

    # --- Model ---
    model_config = AutoConfig.from_pretrained(
        model_args.model_dir_1,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label),
    )

    model = MultimodalDetrMetaCLIP(
        model_name_1=model_args.model_dir_1,
        model_name_2=model_args.model_dir_2,
        config=model_config,
        clip_model_name=model_args.clip_model_name,
    )

    # --- Processors ---
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.base_model_name,
        do_resize=True,
        size={"longest_edge": data_args.image_size, "shortest_edge": data_args.image_size},
        do_pad=True,
    )

    clip_processor = AutoProcessor.from_pretrained(model_args.clip_model_name)

    # --- Datasets ---
    try:
        combined_dataset = prepare_datasets(data_args, image_processor, clip_processor)
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        sys.exit(1)

    # --- Trainer ---
    eval_metrics_fn = partial(
        compute_detection_metrics,
        image_processor=image_processor,
        id2label=id2label,
        threshold=0.5,
    )

    if training_args.do_train and "train" not in combined_dataset:
        raise ValueError("Training is enabled but no 'train' dataset is available.")
    if training_args.do_eval and "validation" not in combined_dataset:
        raise ValueError("Evaluation is enabled but no 'validation' dataset is available.")

    trainer = MultimodalMetaCLIPTrainer(
        fusion_args=fusion_args,
        model=model,
        args=training_args,
        train_dataset=combined_dataset["train"] if training_args.do_train else None,
        eval_dataset=combined_dataset["validation"] if training_args.do_eval else None,
        tokenizer=image_processor,
        compute_metrics=eval_metrics_fn,
        data_collator=collate_fn_multimodal,
    )

    if "wandb" in training_args.report_to:
        try:
            import wandb
            wandb.init(
                project="VLCFusion-ATR", entity="EDCR",
                name=training_args.run_name,
                config=vars(training_args),
            )
        except ImportError:
            logger.warning("wandb not installed. Skipping wandb setup.")

    # --- Train ---
    if training_args.do_train:
        checkpoint_to_resume = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint_to_resume = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint_to_resume = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # --- Evaluate ---
    if training_args.do_eval:
        if "validation" in combined_dataset:
            logger.info("*** Evaluate on Validation Set ***")
            eval_metrics = trainer.evaluate(eval_dataset=combined_dataset["validation"])
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        if "test" in combined_dataset:
            logger.info("*** Evaluate on Test Set ***")
            test_metrics = trainer.evaluate(
                eval_dataset=combined_dataset["test"], metric_key_prefix="test",
            )
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)

    logger.info("Training/evaluation complete.")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
