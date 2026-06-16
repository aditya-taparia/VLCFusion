import logging
import os
import sys
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

# import albumentations as A
from torchvision.transforms import v2
from torchvision import tv_tensors
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry

# Assuming MultimodalDetr is in a local file or installed package
from multimodal_detr import MultimodalDetr # Make sure this import works in your environment

# --- Configuration Dataclasses ---
@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_dir_1: str = field(default="models/45k_seen_ir/checkpoint-24550", metadata={"help": "Path to the first pretrained model component (e.g., IR model checkpoint)."})
    model_dir_2: str = field(default="models/45k_seen_vis/checkpoint-24550", metadata={"help": "Path to the second pretrained model component (e.g., Visible model checkpoint)."})
    base_model_name: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Base DETR model name for image processor and initial config."}
    )
    ensemble_method: str = field(
        default="VLCAM",
        metadata={"help": "Method for ensembling/fusing the multimodal features."}
    )
    num_vlc_blocks: int = field(
        default=2,
        metadata={"help": "Number of stacked VLC blocks in the fusion head (CrossCBAM_DiT_V11 only). Paper default 2; ablation sweeps 2/4/6/8."}
    )
    # Add any other model-specific hyperparameters here

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    visible_dataset_dir: str = field(default="/data/ataparia/Darpa_datasets/atr_dataset/visible_seen", metadata={"help": "Path to the visible spectrum dataset directory (imagefolder structure)."})
    ir_dataset_dir: str = field(default="/data/ataparia/Darpa_datasets/atr_dataset/infrared_seen", metadata={"help": "Path to the infrared spectrum dataset directory (imagefolder structure)."})
    train_conditions_file: str = field(default="conditions/seen/vlm_train.json", metadata={"help": "Path to training conditions JSON."})
    val_conditions_file: str = field(default="conditions/seen/vlm_val.json", metadata={"help": "Path to validation conditions JSON."})
    test_conditions_file: str = field(default="conditions/seen/vlm_test.json", metadata={"help": "Path to test conditions JSON."})
    condition_indices_to_sample_str: Optional[str] = field(
        default="16,13,1,11,15,19,18", # Default from your script
        metadata={"help": "Comma-separated string of 1-based condition indices to sample. 'None' or empty for all."}
    )
    image_size: int = field(default=480, metadata={"help": "Target image size (max height/width) for processing."})
    num_classes: int = field(default=10, metadata={"help": "Number of object classes."}) # Based on your categories_to_tgttype

@dataclass
class FusionTrainingArguments:
    """Arguments for fusion training: per-group LR multiplier for fusion layers only (backbones stay frozen)."""

    fusion_lr_multiplier: float = field(
        default=1.0,
        metadata={"help": "LR multiplier for fusion layers (transform_layer / transform_queries)."},
    )

# --- Global Constants & Mappings (derived from DataArguments or fixed) ---
# This can be initialized in main() after parsing DataArguments
# For example: id2label, label2id

# --- Utility Functions ---
@dataclass
class ModelOutput: # Kept as is, seems fine for compute_metrics
    logits: torch.Tensor
    pred_boxes: torch.Tensor

def load_conditions_data(file_path: str) -> Dict[str, List[Union[bool, float]]]:
    """Loads conditions data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            conditions = json.load(f)
        # Basic validation: ensure it's a dictionary
        if not isinstance(conditions, dict):
            raise ValueError(f"Conditions file {file_path} should contain a JSON object (dictionary).")
        return conditions
    except FileNotFoundError:
        logging.error(f"Conditions file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from conditions file: {file_path}")
        raise

def get_effective_conditions_and_count(
    train_cond_path: str,
    val_cond_path: str,
    test_cond_path: str,
    indices_str: Optional[str]
) -> Tuple[Optional[np.ndarray], int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Loads conditions and determines the number of conditions to use."""
    train_conditions = load_conditions_data(train_cond_path)
    val_conditions = load_conditions_data(val_cond_path)
    test_conditions = load_conditions_data(test_cond_path)

    if not train_conditions:
        raise ValueError("Training conditions could not be loaded or are empty.")
    
    # Determine n_conditions from the first entry in train_conditions
    first_train_entry_key = next(iter(train_conditions))
    n_conditions_total = len(train_conditions[first_train_entry_key])

    indices_to_sample_arr = None
    n_conditions_effective = n_conditions_total

    if indices_str and indices_str.lower() != "none":
        try:
            indices_to_sample_arr = np.array([int(x.strip()) for x in indices_str.split(',') if x.strip()])
            if indices_to_sample_arr.size > 0:
                if np.any(indices_to_sample_arr <= 0):
                    raise ValueError("Condition indices must be 1-based positive integers.")
                # Adjust to be 0-indexed for array slicing later if needed, or keep 1-based for direct use.
                # The original code uses `indices_to_sample - 1`, so we'll prepare for that.
                n_conditions_effective = indices_to_sample_arr.shape[0]
            else: # Empty string after processing
                indices_to_sample_arr = None # Treat as no specific sampling
        except ValueError as e:
            logging.warning(f"Could not parse condition_indices_to_sample_str '{indices_str}': {e}. Using all conditions.")
            indices_to_sample_arr = None
            n_conditions_effective = n_conditions_total
    
    logging.info(f"Total conditions available: {n_conditions_total}")
    logging.info(f"Effective number of conditions to be used: {n_conditions_effective}")
    if indices_to_sample_arr is not None:
        logging.info(f"Using specific condition indices (1-based): {indices_to_sample_arr.tolist()}")

    return indices_to_sample_arr, n_conditions_effective, train_conditions, val_conditions, test_conditions


def format_image_annotations_as_coco(
    image_id: str, categories: List[int], areas: List[float], bboxes: List[Tuple[float, ...]]
) -> Dict[str, Any]:
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id, "category_id": category, "iscrowd": 0,
            "area": area, "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)
    return {"image_id": image_id, "annotations": annotations}

def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    boxes_converted = center_to_corners_format(boxes)
    height, width = image_size
    boxes_converted = boxes_converted * torch.tensor([[width, height, width, height]], device=boxes.device)
    return boxes_converted

# --- Data Transformation ---
# def _process_single_modality_for_transform(
#     image_id: Any, image: Any, objects: Dict[str, Any], transform: A.Compose
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """Helper to apply transform to one image and its annotations."""
#     image_np = np.array(image.convert("RGB"))
#     # Ensure objects["bbox"] is a list of lists/tuples as expected by albumentations
#     bboxes = objects.get("bbox", [])
#     if not all(isinstance(b, (list, tuple)) for b in bboxes):
#         # This might happen if bbox is a single list for a single object.
#         # Albumentations expects a list of bboxes.
#         if isinstance(bboxes, list) and len(bboxes) == 4 and all(isinstance(n, (int,float)) for n in bboxes):
#              bboxes = [bboxes] # Wrap single bbox in a list
#         else: # If it's more complex or truly malformed, log and use empty
#             logging.warning(f"Image {image_id}: Malformed bboxes, attempting to use empty: {bboxes}")
#             bboxes = []


#     categories = objects.get("category", [])
#     # Ensure categories match bboxes length if bboxes exist
#     if len(bboxes) > 0 and len(categories) != len(bboxes):
#         logging.warning(f"Image {image_id}: Mismatch between bbox count ({len(bboxes)}) and category count ({len(categories)}). Using categories up to bbox count or empty.")
#         categories = categories[:len(bboxes)] if categories else [0]*len(bboxes) # Default cat 0 if missing

#     output = transform(image=image_np, bboxes=bboxes, category=categories)
    
#     formatted_annotations = format_image_annotations_as_coco(
#         str(image_id), output["category"], objects.get("area", [0.0] * len(output["bboxes"])), output["bboxes"]
#     ) # Ensure area matches output bbox count
#     return output["image"], formatted_annotations


def augment_and_transform_batch_multimodal(
    examples: Mapping[str, Any],
    # transform: A.Compose,
    transform: v2.Compose,
    image_processor: AutoImageProcessor,
    all_conditions_data: Dict[str, Dict[str, Any]], # Combined conditions {split_name: conditions}
    current_split_name: str, # "train", "validation", or "test"
    indices_to_sample_conditions: Optional[np.ndarray],
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations simultaneously to IR and Visible to maintain alignment."""
    
    ir_images_processed = []
    vis_images_processed = []
    combined_labels = []

    for ir_img, vis_img, ir_objs, ir_id in zip(
        examples["ir_image"], 
        examples["vis_image"], 
        examples["ir_objects"], 
        examples["ir_image_id"]
    ):
        # 1. Force Size Alignment (Critical for Multimodal)
        if ir_img.size != vis_img.size:
            ir_img = ir_img.resize(vis_img.size)

        # 2. Wrap inputs in TVTensors (This tells PyTorch what they are)
        # Convert PIL -> Tensor
        img_ir_tv = tv_tensors.Image(ir_img)
        img_vis_tv = tv_tensors.Image(vis_img)
        
        # Handle Bounding Boxes
        raw_boxes = ir_objs.get("bbox", [])
        if len(raw_boxes) > 0:
            boxes_tv = tv_tensors.BoundingBoxes(
                raw_boxes, 
                format=tv_tensors.BoundingBoxFormat.XYWH, 
                canvas_size=(ir_img.height, ir_img.width)
            )
            labels_tv = torch.tensor(ir_objs.get("category", []), dtype=torch.int64)
        else:
            # Empty placeholders
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), 
                format=tv_tensors.BoundingBoxFormat.XYWH, 
                canvas_size=(ir_img.height, ir_img.width)
            )
            labels_tv = torch.tensor([], dtype=torch.int64)

        # 3. APPLY THE TRANSFORM
        # This one line transforms IR, Vis, and Boxes simultaneously & identically
        output = transform({
            "ir": img_ir_tv,
            "vis": img_vis_tv,
            "boxes": boxes_tv,
            "labels": labels_tv
        })
        
        out_ir = output["ir"]
        out_vis = output["vis"]
        out_boxes = output["boxes"]
        out_labels = output["labels"]

        # 4. Prepare Outputs
        # Add a batch dimension for the image processor if needed, or stack manually
        ir_images_processed.append(out_ir)
        vis_images_processed.append(out_vis)
        
        new_h, new_w = out_ir.shape[-2:]
        
        # Convert boxes from XYWH absolute (COCO) to CXCYWH normalized (DETR)
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
            "orig_size": size_tensor # Important: Update orig_size to new crop size
        }
        combined_labels.append(formatted_ann)

    # 5. Stack into Batch
    # Stack IR and Vis along channel dim (3+3 = 6 channels)
    # Shape: (Batch, 6, H, W)
    pixel_values = torch.cat([
        torch.stack(ir_images_processed), 
        torch.stack(vis_images_processed)
    ], dim=1)
    
    final_batch = {"pixel_values": pixel_values, "labels": combined_labels}

    # Add Conditions (Logic remains the same as your code)
    split_specific_conditions = all_conditions_data[current_split_name]
    for label_dict in final_batch["labels"]:
        img_id_val = label_dict['image_id']
        if hasattr(img_id_val, 'item'):
            image_id_str = str(img_id_val.item())
        else:
            image_id_str = str(img_id_val)
        conditions_for_image = split_specific_conditions.get(image_id_str)
        
        if conditions_for_image is None:
            num_eff_cond = indices_to_sample_conditions.shape[0] if indices_to_sample_conditions is not None else 1
            conditions_tensor = torch.zeros(num_eff_cond, dtype=torch.float)
        else:
            if isinstance(conditions_for_image[0], bool):
                conditions_tensor_full = torch.tensor([1.0 if c else 0.0 for c in conditions_for_image], dtype=torch.float)
            else:
                conditions_tensor_full = torch.tensor(conditions_for_image, dtype=torch.float)
            
            if indices_to_sample_conditions is not None:
                zero_based_indices = torch.from_numpy(indices_to_sample_conditions - 1).long()
                conditions_tensor = conditions_tensor_full[zero_based_indices]
            else:
                conditions_tensor = conditions_tensor_full
        
        label_dict['conditions'] = conditions_tensor.to(pixel_values.device) # Ensure device match if needed later

    if not return_pixel_mask:
        final_batch.pop("pixel_mask", None)

    return final_batch


# --- Collate Function & Metrics ---
def collate_fn_multimodal(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    """Custom collate function for multimodal object detection batch."""
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch] # Labels is a list of dicts, keep as list
    
    collated_batch = {"pixel_values": pixel_values, "labels": labels}
    if "pixel_mask" in batch[0] and batch[0]["pixel_mask"] is not None:
        collated_batch["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return collated_batch

# @torch.no_grad()
# def compute_detection_metrics(
#     evaluation_results: EvalPrediction,
#     image_processor: AutoImageProcessor,
#     id2label: Mapping[int, str],
#     threshold: float = 0.0,
# ) -> Mapping[str, float]:
#     predictions = evaluation_results.predictions
#     targets_raw = evaluation_results.label_ids

#     if isinstance(predictions, tuple):
#         logits_all = predictions[1] 
#         boxes_all = predictions[2]
#     elif isinstance(predictions, dict):
#         logits_all = predictions["logits"]
#         boxes_all = predictions["pred_boxes"]
#     else:
#         # Fallback if structure is different (unlikely given your model code)
#         logits_all = predictions[0]
#         boxes_all = predictions[1]
    
#     processed_predictions, processed_targets = [], []
    
#     # 1. Process Targets
#     for target in targets_raw:
#         # Targets are already a list of dicts, so we iterate directly
#         # Check for orig_size safety (converted to tensor in previous step)
#         if "orig_size" not in target:
#             img_h, img_w = 480, 480
#         else:
#             # It might be a tensor now, so .tolist() or direct access
#             size_t = target["orig_size"]
#             if hasattr(size_t, "tolist"):
#                 img_h, img_w = size_t.tolist()
#             else:
#                 img_h, img_w = size_t

#         boxes = torch.tensor(target["boxes"])
#         # Convert [x,y,w,h] to [x1,y1,x2,y2] absolute
#         boxes = convert_bbox_yolo_to_pascal(boxes, (img_h, img_w))
#         labels = torch.tensor(target["class_labels"])
#         processed_targets.append({"boxes": boxes, "labels": labels})
    
#     # 2. Process Predictions
#     # Now we iterate over the SAMPLES (rows), not the tuple columns
#     for i in range(len(logits_all)):
#         batch_logits = torch.tensor(logits_all[i])
#         batch_boxes = torch.tensor(boxes_all[i])
        
#         # Get target size for this specific image
#         # targets_raw[i] corresponds to the i-th image
#         if "orig_size" in targets_raw[i]:
#             size_val = targets_raw[i]["orig_size"]
#             if hasattr(size_val, "tolist"):
#                 target_size = size_val.tolist()
#             else:
#                 target_size = size_val
#         else:
#             target_size = [480, 480]

#         target_sizes = torch.tensor([target_size]) # Post-processor expects (1, 2) for single item

#         # Wrap for processor
#         output = ModelOutput(logits=batch_logits.unsqueeze(0), pred_boxes=batch_boxes.unsqueeze(0))
        
#         # Post-process
#         post_processed_output = image_processor.post_process_object_detection(
#             output, threshold=threshold, target_sizes=target_sizes
#         )
#         processed_predictions.extend(post_processed_output)
    
#     if not processed_targets or not processed_predictions:
#         logging.warning("No valid targets or predictions to compute metrics.")
#         return {"map": 0.0}


#     # 3. Compute mAP
#     metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
#     metric.update(processed_predictions, processed_targets)
#     metrics = metric.compute()

#     # Format Output
#     final_metrics = {}
#     for k, v in metrics.items():
#         if k != "classes" and k != "map_per_class" and k != "mar_100_per_class":
#              final_metrics[k] = round(v.item(), 4)

#     # Per-class metrics
#     if "map_per_class" in metrics:
#         for class_id, class_map in zip(metrics["classes"], metrics["map_per_class"]):
#             class_name = id2label[class_id.item()] if id2label else str(class_id.item())
#             final_metrics[f"map_{class_name}"] = round(class_map.item(), 4)

#     return final_metrics

@torch.no_grad()
def compute_detection_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    id2label: Mapping[int, str],
    threshold: float = 0.0,
) -> Mapping[str, float]:
    # Because eval_do_concat_batches=False, these are lists of batches
    # Predictions structure: [ (batch1_loss, batch1_logits, batch1_boxes), (batch2...), ... ]
    # Targets structure: [ [img1_dict, img2_dict], [img3_dict...], ... ]
    predictions_batches = evaluation_results.predictions
    targets_batches = evaluation_results.label_ids

    processed_predictions, processed_targets = [], []
    
    # Iterate over batches first (Old Code Logic)
    for batch_pred, batch_targets in zip(predictions_batches, targets_batches):
        
        # 1. Unpack Predictions for this batch
        # Based on your old code logic: Index 1=Logits, Index 2=Boxes
        if isinstance(batch_pred, tuple):
            b_logits = torch.tensor(batch_pred[1])
            b_boxes = torch.tensor(batch_pred[2])
        elif isinstance(batch_pred, dict):
            b_logits = torch.tensor(batch_pred["logits"])
            b_boxes = torch.tensor(batch_pred["pred_boxes"])
        else:
            # Fallback for object-like predictions
            b_logits = torch.tensor(batch_pred.logits)
            b_boxes = torch.tensor(batch_pred.pred_boxes)
            
        # 2. Iterate over images in this batch
        for i, target in enumerate(batch_targets):
            # --- Process Target ---
            # Handle orig_size (it might be a tensor now due to our previous fix)
            if "orig_size" in target:
                sz = target["orig_size"]
                if hasattr(sz, "tolist"):
                    img_h, img_w = sz.tolist()
                else:
                    img_h, img_w = sz
            else:
                img_h, img_w = 480, 480
            
            t_boxes = torch.tensor(target["boxes"])
            # Convert [x,y,w,h] to [x1,y1,x2,y2] absolute
            t_boxes = convert_bbox_yolo_to_pascal(t_boxes, (img_h, img_w))
            t_labels = torch.tensor(target["class_labels"])
            processed_targets.append({"boxes": t_boxes, "labels": t_labels})

            # --- Process Prediction ---
            # Extract the i-th row from the batch tensors
            p_logits = b_logits[i]
            p_boxes = b_boxes[i]
            
            # Post-process expects a batch, so we unsqueeze to make shape (1, ...)
            target_sizes = torch.tensor([[img_h, img_w]])
            output = ModelOutput(logits=p_logits.unsqueeze(0), pred_boxes=p_boxes.unsqueeze(0))
            
            post_processed = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            processed_predictions.extend(post_processed)
    
    if not processed_targets or not processed_predictions:
        logging.warning("No valid targets or predictions to compute metrics.")
        return {"map": 0.0}

    # 3. Compute mAP
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(processed_predictions, processed_targets)
    metrics = metric.compute()

    # Format Output
    final_metrics = {}
    for k, v in metrics.items():
        if k != "classes" and k != "map_per_class" and k != "mar_100_per_class":
             final_metrics[k] = round(v.item(), 4)

    # Per-class metrics
    if "map_per_class" in metrics:
        for class_id, class_map in zip(metrics["classes"], metrics["map_per_class"]):
            class_name = id2label[class_id.item()] if id2label else str(class_id.item())
            final_metrics[f"map_{class_name}"] = round(class_map.item(), 4)

    return final_metrics


def prepare_datasets(
    data_args: DataArguments, 
    image_processor: AutoImageProcessor, 
    all_conditions: Dict[str, Dict[str, Any]], 
    indices_to_sample_conditions: Optional[np.ndarray]
) -> DatasetDict:
    """Loads, preprocesses, and combines visible and IR datasets using original logic."""
    
    # 1. Load Datasets (Assumes metadata.jsonl exists in folders, as per original code)
    logging.info(f"Loading visible dataset from: {data_args.visible_dataset_dir}")
    vis_custom_data = load_dataset("imagefolder", data_dir=data_args.visible_dataset_dir)
    
    logging.info(f"Loading IR dataset from: {data_args.ir_dataset_dir}")
    ir_custom_data = load_dataset("imagefolder", data_dir=data_args.ir_dataset_dir)

    # 2. Sort and Shuffle (CRITICAL for alignment)
    # We must ensure 'image_id' exists. If load_dataset didn't find it in metadata,
    # we assume file_names align.
    
    def ensure_alignment_columns(dataset, prefix):
        # If image_id is missing, use filename (without extension) as sort key
        if "image_id" not in dataset["train"].column_names:
            def add_filename_id(example):
                return {"image_id": os.path.splitext(os.path.basename(example["image"].filename))[0]}
            dataset = dataset.map(add_filename_id)
        
        # Sort and Shuffle with fixed seed
        for split in dataset.keys():
            dataset[split] = dataset[split].sort("image_id").shuffle(seed=42)
            
            # Rename columns to avoid collision during concatenation
            # We rename EVERYTHING except 'image_id' used for sorting check (optional)
            # but usually we just prefix everything.
            dataset[split] = dataset[split].rename_columns(
                {col: f"{prefix}_{col}" for col in dataset[split].column_names}
            )
        return dataset

    vis_custom_data = ensure_alignment_columns(vis_custom_data, "vis")
    ir_custom_data = ensure_alignment_columns(ir_custom_data, "ir")

    # 3. Concatenate
    processed_splits = {}
    for split in ["train", "validation", "test"]:
        if split not in vis_custom_data or split not in ir_custom_data:
            continue
            
        # Verify alignment (sanity check)
        if len(vis_custom_data[split]) != len(ir_custom_data[split]):
             logging.warning(f"Length mismatch in {split}: Vis {len(vis_custom_data[split])} vs IR {len(ir_custom_data[split])}")

        combined_split = concatenate_datasets([vis_custom_data[split], ir_custom_data[split]], axis=1)
        processed_splits[split] = combined_split

    combined_dataset = DatasetDict(processed_splits)

    # 4. Define Transforms (Same as your modified code)
    additional_targets = {'image_vis': 'image'}
    
    # train_alb_transform = A.Compose([
    #     A.Compose([
    #         A.SmallestMaxSize(max_size=data_args.image_size, p=1.0),
    #         A.RandomSizedBBoxSafeCrop(height=data_args.image_size, width=data_args.image_size, p=1.0),
    #     ], p=0.2),
    #     A.OneOf([
    #         A.Blur(blur_limit=7, p=0.5), A.MotionBlur(blur_limit=7, p=0.5),
    #         A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
    #     ], p=0.1),
    #     A.Perspective(p=0.1), A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.1),
    # ], 
    #     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1.0, min_visibility=0.3), 
    #     additional_targets=additional_targets)

    # eval_alb_transform = A.Compose(
    #     [A.NoOp()], 
    #     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True), 
    #     additional_targets=additional_targets
    # )
    
    # Define the complex training transform
    train_torch_transform = v2.Compose([
        # --- Resize & Crop (p=0.2) ---
        # Albumentations: SmallestMaxSize -> RandomSizedBBoxSafeCrop
        # Torchvision Equivalent: Resize short edge -> RandomCrop
        v2.RandomChoice([
            # Option A: Random Crop (20% chance based on your logic, but scaled to choice)
            v2.Compose([
                v2.Resize(size=data_args.image_size, antialias=True), # Resize short edge to 480
                v2.RandomCrop(size=(data_args.image_size, data_args.image_size), pad_if_needed=True) # Crop to 480x480
            ]),
            # Option B: Standard Resize (80% chance) - Force to square
            v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True) 
        ], p=[0.2, 0.8]),

        # --- Blurs (OneOf p=0.1) ---
        # Albumentations: Blur, MotionBlur, Defocus
        # Torchvision: We use GaussianBlur with different sigmas to approximate
        v2.RandomApply([
            v2.RandomChoice([
                v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)), # Standard Blur
                v2.GaussianBlur(kernel_size=7, sigma=(2.0, 5.0)), # Stronger (approx Motion)
            ])
        ], p=0.1),

        # --- Geometry ---
        v2.RandomPerspective(distortion_scale=0.1, p=0.1),
        v2.RandomHorizontalFlip(p=0.5),

        # --- Color (Brightness/Contrast p=0.5) ---
        v2.RandomApply([
            v2.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.5),

        # --- Color (Hue/Sat p=0.1) ---
        v2.RandomApply([
            v2.ColorJitter(hue=0.1, saturation=0.1)
        ], p=0.1),

        # --- Final Formatting ---
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(),
    ])

    # Define the eval transform (Just formatting)
    eval_torch_transform = v2.Compose([
        v2.Resize(size=(data_args.image_size, data_args.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Apply Transforms
    transform_kwargs_base = {
        "image_processor": image_processor,
        "all_conditions_data": all_conditions,
        "indices_to_sample_conditions": indices_to_sample_conditions,
    }
    
    combined_dataset["train"] = combined_dataset["train"].with_transform(
        partial(augment_and_transform_batch_multimodal, transform=train_torch_transform, current_split_name="train", **transform_kwargs_base)
    )
    if "validation" in combined_dataset:
        combined_dataset["validation"] = combined_dataset["validation"].with_transform(
            partial(augment_and_transform_batch_multimodal, transform=eval_torch_transform, current_split_name="validation", **transform_kwargs_base)
        )
    if "test" in combined_dataset:
        combined_dataset["test"] = combined_dataset["test"].with_transform(
            partial(augment_and_transform_batch_multimodal, transform=eval_torch_transform, current_split_name="test", **transform_kwargs_base)
        )
        
    return combined_dataset

# --- Fusion Training (fusion LR multiplier only; backbones stay frozen) ---

class MultimodalTrainer(Trainer):
    """Trainer subclass that creates separate optimizer param-groups for
    fusion vs other parameters when fusion_lr_multiplier != 1.0.
    Backbone parameters are never included (stay frozen).
    """

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
        backbone_prefixes = (
            "backbone_ir", "input_projection_ir",
            "backbone_rgb", "input_projection_rgb",
        )
        fusion_prefixes = ("transform_layer", "transform_queries")

        groups: Dict[str, list] = {
            "fusion_decay": [],   "fusion_no_decay": [],
            "other_decay": [],    "other_no_decay": [],
        }

        for name, param in model.named_parameters():
            is_backbone = any(name.startswith(p) for p in backbone_prefixes)
            is_fusion = any(name.startswith(p) for p in fusion_prefixes)

            if is_backbone:
                continue  # never train backbone
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
            optimizer_grouped.append({"params": groups["fusion_decay"],      "lr": fusion_lr,  "weight_decay": wd,  "group_name": "fusion"})
        if groups["fusion_no_decay"]:
            optimizer_grouped.append({"params": groups["fusion_no_decay"],   "lr": fusion_lr,  "weight_decay": 0.0, "group_name": "fusion"})
        if groups["other_decay"]:
            optimizer_grouped.append({"params": groups["other_decay"],       "lr": base_lr,    "weight_decay": wd,  "group_name": "other"})
        if groups["other_no_decay"]:
            optimizer_grouped.append({"params": groups["other_no_decay"],    "lr": base_lr,    "weight_decay": 0.0, "group_name": "other"})

        self.optimizer = torch.optim.AdamW(optimizer_grouped)
        return self.optimizer


# --- Main Function ---
def main():
    default_args = [
        "--output_dir", "model/45k_cbam_fusion_7_cond_fusion",
        "--run_name", "45k_cbam_fusion_7_cond_fusion",
        
        # Training Strategy
        "--num_train_epochs", "100",
        "--per_device_train_batch_size", "128",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "5e-5",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--weight_decay", "1e-4",
        "--max_grad_norm", "1.0",
        # "--fp16", "False", # Removed: Default is already False. Adding "False" might break parsing.
        
        # Evaluation & Saving
        "--metric_for_best_model", "eval_map",
        "--greater_is_better", "True",
        "--load_best_model_at_end", "True",
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--save_total_limit", "5",
        "--logging_steps", "100",
        "--report_to", "wandb",
        
        # Data & Model Processing
        "--remove_unused_columns", "False",
        "--eval_do_concat_batches", "False",
        "--do_train", # Presence of flag = True
        "--do_eval",  # Presence of flag = True
        
        "--dataloader_num_workers", "16",  # This is the most important setting
        "--dataloader_pin_memory", "True",
    ]
    
    import sys
    combined_args = default_args + sys.argv[1:]
    
    parser = HfArgumentParser((ModelArguments, DataArguments, FusionTrainingArguments, TrainingArguments))
    model_args, data_args, fusion_args, training_args = parser.parse_args_into_dataclasses(args=combined_args)

    # Setup logging
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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Fusion training parameters {fusion_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load conditions and determine n_conditions
    indices_to_sample_arr, n_conditions_eff, train_cond, val_cond, test_cond = \
        get_effective_conditions_and_count(data_args.train_conditions_file, data_args.val_conditions_file, data_args.test_conditions_file, data_args.condition_indices_to_sample_str)
    
    all_conditions_for_transform = {
        "train": train_cond,
        "validation": val_cond,
        "test": test_cond
    }

    # Define id2label, label2id (consistent with data and model config)
    # Based on your original script's categories_to_tgttype
    _categories_to_tgttype = {
        0: 'PICKUP', 1: 'SUV', 2: 'BTR70', 3: 'BRDM2', 4: 'BMP2',
        5: 'T72', 6: 'ZSU23', 7: '2S3', 8: 'MTLB', 9: 'D20',
    }
    # Filter by num_classes if data_args.num_classes is less than the full map
    id2label = {k: v for k, v in _categories_to_tgttype.items() if k < data_args.num_classes}
    if len(id2label) != data_args.num_classes:
        logger.warning(f"Mismatch between num_classes ({data_args.num_classes}) and derived id2label size ({len(id2label)}). Adjust num_classes or mapping.")
    label2id = {v: k for k, v in id2label.items()}


    # Initialize model configuration
    model_config = AutoConfig.from_pretrained(
        model_args.model_dir_1, # Assuming this dir contains a valid config.json for DETR
        label2id=label2id,
        id2label=id2label,
        num_labels=len(id2label), # Ensure num_labels is consistent
    )
    
    # Initialize model
    # Ensure MultimodalDetr class can accept n_conditions
    model = MultimodalDetr(
        model_name_1=model_args.model_dir_1,
        model_name_2=model_args.model_dir_2,
        config=model_config, # Pass the updated config
        ensemble_method=model_args.ensemble_method,
        n_conditions=n_conditions_eff, # Pass the effective number of conditions
        num_vlc_blocks=model_args.num_vlc_blocks
    )

    # Initialize image processor
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.base_model_name, # Use base DETR model for processor
        do_resize=True,
        size={"longest_edge": data_args.image_size, "shortest_edge": data_args.image_size}, # Adjusted to typical DETR sizing
        # size={"max_height": data_args.image_size, "max_width": data_args.image_size}, # Original
        do_pad=True, # DETR typically pads to square
        # pad_size={"height": data_args.image_size, "width": data_args.image_size} # Original
    )
    # For DETR, padding to a square image (e.g., data_args.image_size x data_args.image_size) is common.
    # The image_processor call might need `size` as int (for square) or dict `{"height": H, "width": W}`.
    # If using `pad_to_multiple_of` or similar, ensure consistency.


    # Prepare datasets
    try:
        combined_dataset = prepare_datasets(data_args, image_processor, all_conditions_for_transform, indices_to_sample_arr)
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        logger.error("Please ensure your dataset loading logic in 'prepare_datasets' correctly loads images and object detection annotations (image_id, objects with bbox, category, area). The default 'imagefolder' loader is usually for classification.")
        sys.exit(1)


    # Initialize Trainer
    eval_metrics_fn_with_args = partial(
        compute_detection_metrics,
        image_processor=image_processor,
        id2label=id2label,
        threshold=0.5, # From your original script
    )

    if training_args.do_train and "train" not in combined_dataset:
        raise ValueError("Training is enabled but no 'train' dataset is available.")
    if training_args.do_eval and "validation" not in combined_dataset:
        raise ValueError("Evaluation is enabled but no 'validation' dataset is available.")

    trainer = MultimodalTrainer(
        fusion_args=fusion_args,
        model=model,
        args=training_args,
        train_dataset=combined_dataset["train"] if training_args.do_train else None,
        eval_dataset=combined_dataset["validation"] if training_args.do_eval else None,
        tokenizer=image_processor, # For some internal HuggingFace checks
        compute_metrics=eval_metrics_fn_with_args,
        data_collator=collate_fn_multimodal,
    )

    # WandB setup (from your script)
    if "wandb" in training_args.report_to:
        try:
            import wandb
            wandb.init(project="VLCFusion-ATR", entity="EDCR", name=training_args.run_name, config=vars(training_args)) # Add model_args, data_args to config if desired
        except ImportError:
            logger.warning("wandb report_to specified but wandb is not installed. Skipping wandb setup.")


    # Training
    if training_args.do_train:
        checkpoint_to_resume = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint_to_resume = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint_to_resume = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        trainer.save_model() # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if "validation" in combined_dataset:
            logger.info("*** Evaluate on Validation Set ***")
            eval_metrics = trainer.evaluate(eval_dataset=combined_dataset["validation"])
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        
        if "test" in combined_dataset: # Optional: evaluate on test set if available
            logger.info("*** Evaluate on Test Set ***")
            test_metrics = trainer.evaluate(eval_dataset=combined_dataset["test"], metric_key_prefix="test")
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)

    logger.info("Training/evaluation complete.")

if __name__ == "__main__":
    # It's good practice to set CUDA_VISIBLE_DEVICES externally:
    # Example: CUDA_VISIBLE_DEVICES=1 python your_script_name.py ...
    # If you must set it in script, do it before any torch imports if possible,
    # but external is preferred for flexibility.
    # Respect an externally-provided CUDA_VISIBLE_DEVICES; only default if unset.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    main()