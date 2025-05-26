import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Mapping, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from multimodal_detr import MultimodalDetr

# Read conditions files
import json
with open("conditions/seen/vlm_train.json") as f:
    train_conditions = json.load(f)
with open("conditions/seen/vlm_val.json") as f:
    val_conditions = json.load(f)
with open("conditions/seen/vlm_test.json") as f:
    test_conditions = json.load(f)

# indices_to_sample = None
indices_to_sample = np.array([16, 13, 1, 11, 15, 19, 18])
n_conditions = len(train_conditions["0"])

if indices_to_sample is not None:
    if indices_to_sample.shape[0] < n_conditions:
        n_conditions = indices_to_sample.shape[0]

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
    mode: str = "train",
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""
    
    images = []
    annotations = []
  
    for image_id, image, objects in zip(examples["ir_image_id"], examples["ir_image"], examples["ir_objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)
    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    
    vis_images = []
    vis_annotations = []
    for image_id, image, objects in zip(examples["vis_image_id"], examples["vis_image"], examples["vis_objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        vis_images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        vis_annotations.append(formatted_annotations)
    # Apply the image processor transformations: resizing, rescaling, normalization
    vis_result = image_processor(images=vis_images, annotations=vis_annotations, return_tensors="pt")
    
    # Extract pixel_values from vis_result
    vis_pixel_values = vis_result.pop("pixel_values")
    
    # Concatenate the pixel values from vis_result and result and make it 6 channels
    result["pixel_values"] = torch.cat([result["pixel_values"], vis_pixel_values], dim=1)
    
    if mode == "train":
        conditions_data = train_conditions
    elif mode == "validation":
        conditions_data = val_conditions
    elif mode == "test":
        conditions_data = test_conditions
    
    for label in result["labels"]:
        conditions = conditions_data[str(label['image_id'].item())]
        
        # Check if the conditions are boolean values
        if isinstance(conditions[0], bool):
            # print("Conditions are boolean values")
            conditions_tensor = torch.tensor([1.0 if c else 0.0 for c in conditions], dtype=torch.float, device=label['image_id'].device)
        else:
            conditions_tensor = torch.tensor(conditions, dtype=torch.float, device=label['image_id'].device)
        # conditions_tensor = torch.tensor(conditions, dtype=torch.float, device=label['image_id'].device)
        # conditions_tensor = torch.tensor([1.0 if c else 0.0 for c in conditions], dtype=torch.float, device=label['image_id'].device)
        
        if indices_to_sample is not None:
            conditions_tensor = conditions_tensor[indices_to_sample - 1]
        label['conditions'] = conditions_tensor
    
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    
    return result

def collate_fn(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


vis_custom_data = load_dataset("imagefolder", data_dir="/mnt/data/ataparia/darpa/visible dataset seen")
vis_custom_data = vis_custom_data.sort(column_names=["image_id"]).shuffle(seed=42)

ir_custom_data = load_dataset("imagefolder", data_dir="/mnt/data/ataparia/darpa/mwir dataset seen")
ir_custom_data = ir_custom_data.sort(column_names=["image_id"]).shuffle(seed=42)

prefix = "vis"
vis_updated_columns = vis_custom_data.copy()
for col in vis_custom_data["train"].column_names:
    vis_updated_columns["train"] = vis_updated_columns["train"].rename_column(col, f"{prefix}_{col}")
for col in vis_updated_columns["validation"].column_names:
    vis_updated_columns["validation"] = vis_updated_columns["validation"].rename_column(col, f"{prefix}_{col}")
for col in vis_updated_columns["test"].column_names:
    vis_updated_columns["test"] = vis_updated_columns["test"].rename_column(col, f"{prefix}_{col}")

prefix = "ir"
ir_updated_columns = ir_custom_data.copy()
for col in ir_custom_data["train"].column_names:
    ir_updated_columns["train"] = ir_updated_columns["train"].rename_column(col, f"{prefix}_{col}")
for col in ir_updated_columns["validation"].column_names:
    ir_updated_columns["validation"] = ir_updated_columns["validation"].rename_column(col, f"{prefix}_{col}")
for col in ir_updated_columns["test"].column_names:
    ir_updated_columns["test"] = ir_updated_columns["test"].rename_column(col, f"{prefix}_{col}")

combine_train = concatenate_datasets([vis_updated_columns["train"], ir_updated_columns["train"]], axis=1)
combine_validation = concatenate_datasets([vis_updated_columns["validation"], ir_updated_columns["validation"]], axis=1)
combine_test = concatenate_datasets([vis_updated_columns["test"], ir_updated_columns["test"]], axis=1)

combined_dataset = DatasetDict({
    "train": combine_train,
    "validation": combine_validation,
    "test": combine_test
})

model_dir_1 = "model/45k_seen_ir/checkpoint-24550"
model_dir_2 = "model/45k_seen_vis/checkpoint-24550"
MODEL_NAME = "facebook/detr-resnet-50"

IMAGE_SIZE = 480

categories_to_tgttype = {
    0: 'PICKUP',
    1: 'SUV',
    2: 'BTR70',
    3: 'BRDM2',
    4: 'BMP2',
    # 5: 'T62', # Removing this class since we don't have any samples for this in dataset
    5: 'T72',
    6: 'ZSU23',
    7: '2S3',
    8: 'MTLB',
    9: 'D20',
}

id2label = categories_to_tgttype
label2id = {v: k for k, v in id2label.items()}

config = AutoConfig.from_pretrained(
    model_dir_1,
    label2id=label2id,
    id2label=id2label,
)
model = MultimodalDetr(model_dir_1, model_dir_2, config=config, ensemble_method="CBAM_FiLM", n_conditions=n_conditions)

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size={"max_height": IMAGE_SIZE, "max_width": IMAGE_SIZE},
    do_pad=True,
    pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
)


train_augment_and_transform = A.Compose(
    [
        A.Compose([
            A.SmallestMaxSize(max_size=IMAGE_SIZE, p=1.0),
            A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.Blur(blur_limit=7, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
        ], p=0.1),
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1.0, min_visibility=0.3)
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)

# Make transform functions for batch and apply dataset splits
train_transform_batch = partial(
    augment_and_transform_batch,
    transform=train_augment_and_transform,
    image_processor=image_processor,
    mode="train",
)

validation_transform_batch = partial(
    augment_and_transform_batch,
    transform=validation_transform,
    image_processor=image_processor,
    mode="validation",
)

test_transform_batch = partial(
    augment_and_transform_batch,
    transform=validation_transform,
    image_processor=image_processor,
    mode="test",
)


combined_dataset["train"] = combined_dataset["train"].with_transform(train_transform_batch)
combined_dataset["validation"] = combined_dataset["validation"].with_transform(validation_transform_batch)
combined_dataset["test"] = combined_dataset["test"].with_transform(test_transform_batch)


eval_compute_metrics_fn = partial(
    compute_metrics,
    image_processor=image_processor,
    threshold=0.5, # Evaluate on high confidence
    id2label=id2label,
)

import wandb
wandb.init(project="ECAI-ATR", entity="EDCR")


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="model/45k_cbam_fusion_7_cond_fusion",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    fp16=False,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="wandb",
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=100,
    run_name="45k_cbam_fusion_7_cond_fusion",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset["train"],
    eval_dataset=combined_dataset["validation"],
    tokenizer=image_processor,
    compute_metrics=eval_compute_metrics_fn,
    data_collator=collate_fn,
)

trainer.train(resume_from_checkpoint=False)