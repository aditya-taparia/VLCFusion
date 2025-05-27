import logging
import os
import sys
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import albumentations as A
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
    model_dir_1: str = field(metadata={"help": "Path to the first pretrained model component (e.g., IR model checkpoint)."})
    model_dir_2: str = field(metadata={"help": "Path to the second pretrained model component (e.g., Visible model checkpoint)."})
    base_model_name: str = field(
        default="facebook/detr-resnet-50",
        metadata={"help": "Base DETR model name for image processor and initial config."}
    )
    ensemble_method: str = field(
        default="CBAM_FiLM",
        metadata={"help": "Method for ensembling/fusing the multimodal features."}
    )
    # Add any other model-specific hyperparameters here

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    visible_dataset_dir: str = field(metadata={"help": "Path to the visible spectrum dataset directory (imagefolder structure)."})
    ir_dataset_dir: str = field(metadata={"help": "Path to the infrared spectrum dataset directory (imagefolder structure)."})
    train_conditions_file: str = field(default="conditions/seen/vlm_train.json", metadata={"help": "Path to training conditions JSON."})
    val_conditions_file: str = field(default="conditions/seen/vlm_val.json", metadata={"help": "Path to validation conditions JSON."})
    test_conditions_file: str = field(default="conditions/seen/vlm_test.json", metadata={"help": "Path to test conditions JSON."})
    condition_indices_to_sample_str: Optional[str] = field(
        default="16,13,1,11,15,19,18", # Default from your script
        metadata={"help": "Comma-separated string of 1-based condition indices to sample. 'None' or empty for all."}
    )
    image_size: int = field(default=480, metadata={"help": "Target image size (max height/width) for processing."})
    num_classes: int = field(default=10, metadata={"help": "Number of object classes."}) # Based on your categories_to_tgttype

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
def _process_single_modality_for_transform(
    image_id: Any, image: Any, objects: Dict[str, Any], transform: A.Compose
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Helper to apply transform to one image and its annotations."""
    image_np = np.array(image.convert("RGB"))
    # Ensure objects["bbox"] is a list of lists/tuples as expected by albumentations
    bboxes = objects.get("bbox", [])
    if not all(isinstance(b, (list, tuple)) for b in bboxes):
        # This might happen if bbox is a single list for a single object.
        # Albumentations expects a list of bboxes.
        if isinstance(bboxes, list) and len(bboxes) == 4 and all(isinstance(n, (int,float)) for n in bboxes):
             bboxes = [bboxes] # Wrap single bbox in a list
        else: # If it's more complex or truly malformed, log and use empty
            logging.warning(f"Image {image_id}: Malformed bboxes, attempting to use empty: {bboxes}")
            bboxes = []


    categories = objects.get("category", [])
    # Ensure categories match bboxes length if bboxes exist
    if len(bboxes) > 0 and len(categories) != len(bboxes):
        logging.warning(f"Image {image_id}: Mismatch between bbox count ({len(bboxes)}) and category count ({len(categories)}). Using categories up to bbox count or empty.")
        categories = categories[:len(bboxes)] if categories else [0]*len(bboxes) # Default cat 0 if missing

    output = transform(image=image_np, bboxes=bboxes, category=categories)
    
    formatted_annotations = format_image_annotations_as_coco(
        str(image_id), output["category"], objects.get("area", [0.0] * len(output["bboxes"])), output["bboxes"]
    ) # Ensure area matches output bbox count
    return output["image"], formatted_annotations


def augment_and_transform_batch_multimodal(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    all_conditions_data: Dict[str, Dict[str, Any]], # Combined conditions {split_name: conditions}
    current_split_name: str, # "train", "validation", or "test"
    indices_to_sample_conditions: Optional[np.ndarray],
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations for multimodal object detection."""
    
    ir_images_processed, ir_annotations_processed = [], []
    for img_id, img, objs in zip(examples["ir_image_id"], examples["ir_image"], examples["ir_objects"]):
        proc_img, proc_ann = _process_single_modality_for_transform(img_id, img, objs, transform)
        ir_images_processed.append(proc_img)
        ir_annotations_processed.append(proc_ann)
    
    ir_transformed_batch = image_processor(
        images=ir_images_processed, annotations=ir_annotations_processed, return_tensors="pt"
    )

    vis_images_processed, vis_annotations_processed = [], []
    for img_id, img, objs in zip(examples["vis_image_id"], examples["vis_image"], examples["vis_objects"]):
        proc_img, proc_ann = _process_single_modality_for_transform(img_id, img, objs, transform)
        vis_images_processed.append(proc_img)
        vis_annotations_processed.append(proc_ann)

    vis_transformed_batch = image_processor(
        images=vis_images_processed, annotations=vis_annotations_processed, return_tensors="pt"
    )

    # Combine pixel values (IR first, then Visible)
    ir_pixel_values = ir_transformed_batch.pop("pixel_values")
    vis_pixel_values = vis_transformed_batch.pop("pixel_values")
    
    # The model expects concatenated channels.
    # Assuming ir_pixel_values and vis_pixel_values are [batch, 3, H, W]
    # Resulting pixel_values will be [batch, 6, H, W]
    combined_pixel_values = torch.cat([ir_pixel_values, vis_pixel_values], dim=1)
    
    # Start with IR batch results as the base, then add combined pixels and conditions
    final_batch_result = ir_transformed_batch 
    final_batch_result["pixel_values"] = combined_pixel_values

    # Add conditions to labels
    # Conditions data should be for the current split (train/val/test)
    split_specific_conditions = all_conditions_data[current_split_name]

    for label_dict in final_batch_result["labels"]:
        image_id_tensor = label_dict['image_id']
        image_id_str = str(image_id_tensor.item()) # image_id in conditions is string
        
        conditions_for_image = split_specific_conditions.get(image_id_str)
        if conditions_for_image is None:
            logging.warning(f"No conditions found for image_id {image_id_str} in {current_split_name} split. Using zeros.")
            # Determine number of conditions from one valid entry or config
            num_eff_cond = indices_to_sample_conditions.shape[0] if indices_to_sample_conditions is not None else len(next(iter(split_specific_conditions.values())))
            conditions_tensor = torch.zeros(num_eff_cond, dtype=torch.float, device=image_id_tensor.device)

        else:
            # Convert boolean list to float tensor (0.0 or 1.0)
            if isinstance(conditions_for_image[0], bool):
                conditions_tensor_full = torch.tensor(
                    [1.0 if c else 0.0 for c in conditions_for_image],
                    dtype=torch.float, device=image_id_tensor.device
                )
            else: # Assuming already numbers if not bool
                conditions_tensor_full = torch.tensor(
                    conditions_for_image, dtype=torch.float, device=image_id_tensor.device
                )
            
            if indices_to_sample_conditions is not None:
                # Ensure indices are 0-based for PyTorch tensor indexing
                zero_based_indices = torch.from_numpy(indices_to_sample_conditions - 1).long()
                conditions_tensor = conditions_tensor_full[zero_based_indices]
            else:
                conditions_tensor = conditions_tensor_full
        
        label_dict['conditions'] = conditions_tensor
    
    if not return_pixel_mask:
        final_batch_result.pop("pixel_mask", None)
        
    return final_batch_result


# --- Collate Function & Metrics ---
def collate_fn_multimodal(batch: List[BatchFeature]) -> Mapping[str, Union[torch.Tensor, List[Any]]]:
    """Custom collate function for multimodal object detection batch."""
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = [x["labels"] for x in batch] # Labels is a list of dicts, keep as list
    
    collated_batch = {"pixel_values": pixel_values, "labels": labels}
    if "pixel_mask" in batch[0] and batch[0]["pixel_mask"] is not None:
        collated_batch["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return collated_batch

@torch.no_grad()
def compute_detection_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    id2label: Mapping[int, str],
    threshold: float = 0.0,
) -> Mapping[str, float]:
    predictions, targets_raw = evaluation_results.predictions, evaluation_results.label_ids
    
    # predictions shape: (batch_size, num_queries, num_classes + 1) for logits, (batch_size, num_queries, 4) for boxes
    # We expect predictions to be a tuple/list: (loss_dict, logits, boxes) or similar output from MultimodalDetr
    # The original script indexes batch[1] and batch[2]
    # Assuming predictions is a list where each element is [loss_info_or_None, logits_tensor, boxes_tensor]

    processed_predictions, processed_targets = [], []
    
    # Assuming targets_raw is a list of lists of dictionaries (batch of labels)
    for batch_targets in targets_raw:
        for target in batch_targets: # Each target is a dict for one image
            # orig_size should be in the target dict if image_processor added it
            # This was how the original script got target_sizes
            if "orig_size" not in target:
                 logging.error("orig_size missing from target labels. Cannot compute metrics accurately.")
                 # Fallback or raise error - for now, skip this target for metric calculation
                 continue
            
            img_h, img_w = target["orig_size"]
            # Convert YOLO [0,1] to Pascal VOC absolute [xmin, ymin, xmax, ymax]
            boxes = convert_bbox_yolo_to_pascal(torch.tensor(target["boxes"]), (img_h, img_w))
            labels = torch.tensor(target["class_labels"])
            processed_targets.append({"boxes": boxes, "labels": labels})

    # Assuming 'predictions' is a list where each element corresponds to a batch's output
    # And each batch output is a tuple/list like (None, logits, boxes) as implied by original script
    for batch_prediction_output, batch_targets in zip(predictions, targets_raw):
        # The model output needs to be structured as ModelOutput for post_process_object_detection
        # Original script: batch_logits, batch_boxes = batch[1], batch[2]
        # Ensure batch_prediction_output has this structure
        if len(batch_prediction_output) < 3: # Check based on original indexing
            logging.error("Unexpected prediction output structure. Skipping batch for metrics.")
            continue
        
        logits_np, boxes_np = batch_prediction_output[1], batch_prediction_output[2]
        model_output_for_postproc = ModelOutput(logits=torch.from_numpy(logits_np), pred_boxes=torch.from_numpy(boxes_np))
        
        target_sizes_for_batch = torch.tensor([t["orig_size"] for t in batch_targets if "orig_size" in t])
        if target_sizes_for_batch.ndim == 1: # If only one image in batch and target_sizes is [H,W]
            target_sizes_for_batch = target_sizes_for_batch.unsqueeze(0)
        
        if target_sizes_for_batch.size(0) == 0 and model_output_for_postproc.logits.size(0) > 0 :
            # This can happen if all targets for this batch were skipped due to missing orig_size
            logging.warning("No valid target sizes for a batch with predictions. Using inferred or placeholder sizes.")
            # This is problematic. For now, we might have to skip post-processing this batch's predictions
            # or try to infer target_sizes if possible (e.g., from image_processor if fixed size was used).
            # A robust solution depends on how images are padded/resized.
            # For now, let's assume fixed size if not available, though this is not ideal.
            # Example: if all images are padded to data_args.image_size
            # target_sizes_for_batch = torch.tensor([[data_args.image_size, data_args.image_size]] * model_output_for_postproc.logits.size(0))
            # This is a HACK and needs verification based on actual preprocessing
            continue # Skip this batch if target sizes are missing.

        post_processed = image_processor.post_process_object_detection(
            outputs=model_output_for_postproc,
            threshold=threshold,
            target_sizes=target_sizes_for_batch
        )
        processed_predictions.extend(post_processed)
    
    if not processed_targets or not processed_predictions:
        logging.warning("No valid targets or predictions to compute metrics.")
        return {"map": 0.0, "map_50": 0.0, "map_75": 0.0, "map_small": 0.0, "map_medium": 0.0, "map_large": 0.0}


    metric_calculator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric_calculator.update(processed_predictions, processed_targets)
    computed_metrics = metric_calculator.compute()

    final_metrics = {}
    for k, v in computed_metrics.items():
        if k == "classes" or k.endswith("_per_class"):
            continue
        final_metrics[k] = round(v.item(), 4)

    if "map_per_class" in computed_metrics and "classes" in computed_metrics:
        classes_ids = computed_metrics["classes"].tolist()
        map_per_class_values = computed_metrics["map_per_class"].tolist()
        mar_100_per_class_values = computed_metrics.get("mar_100_per_class", torch.zeros_like(computed_metrics["classes"])).tolist()

        for class_id, class_map, class_mar in zip(classes_ids, map_per_class_values, mar_100_per_class_values):
            class_name = id2label.get(class_id, f"class_{class_id}")
            final_metrics[f"map_{class_name}"] = round(class_map, 4)
            final_metrics[f"mar_100_{class_name}"] = round(class_mar, 4)
            
    return final_metrics


# --- Dataset Loading and Preparation Function ---
def prepare_datasets(
    data_args: DataArguments, 
    image_processor: AutoImageProcessor, 
    all_conditions: Dict[str, Dict[str, Any]], # Combined {split_name: conditions_dict}
    indices_to_sample_conditions: Optional[np.ndarray]
) -> DatasetDict:
    """Loads, preprocesses, and combines visible and IR datasets."""
    
    # Define id2label and label2id based on num_classes or a fixed mapping
    # Using fixed mapping from your script for now
    categories_to_tgttype = {
        0: 'PICKUP', 1: 'SUV', 2: 'BTR70', 3: 'BRDM2', 4: 'BMP2',
        5: 'T72', 6: 'ZSU23', 7: '2S3', 8: 'MTLB', 9: 'D20',
    } # Assuming T62 is indeed excluded as per your original comment
    
    # This part needs to be consistent with how model config is created
    # id2label = {k: v for k, v in categories_to_tgttype.items() if k < data_args.num_classes}
    # label2id = {v: k for k, v in id2label.items()}

    logging.info(f"Loading visible dataset from: {data_args.visible_dataset_dir}")
    vis_custom_data = load_dataset("imagefolder", data_dir=data_args.visible_dataset_dir, drop_labels=True)
    # After load_dataset("imagefolder", ...), you need to load your actual annotations (objects)
    # This part is critical and was missing. Imagefolder only gives 'image'.
    # You need to load metadata.jsonl and merge it.
    # For now, I'll assume the 'objects' column is somehow present or added in a subsequent step
    # that was abstracted in the original script. This needs to be fixed for a runnable script.
    # Placeholder: Manually add `image_id` and `objects` if not present.
    # This is a major simplification and needs proper implementation based on your S4 output.
    def add_metadata_placeholder(examples, idx, prefix):
        # This function's logic depends heavily on how S4 saves data and how `imagefolder` interacts.
        # Let's assume S4's output `metadata.jsonl` is somehow loaded and merged.
        # The simplest way is to load from JSONL directly instead of "imagefolder"
        # For now, just ensure the columns exist for the transform function.
        examples[f"{prefix}_image_id"] = [f"{prefix}_{i}" for i in idx] # Dummy image_id
        examples[f"{prefix}_objects"] = [{"bbox": [], "category": [], "area": []} for _ in idx] # Dummy objects
        return examples

    # This is a placeholder section. You should replace `load_dataset("imagefolder", ...)`
    # with a proper loading mechanism that loads images AND your `metadata.jsonl` annotations.
    # Example using `load_dataset("json", ...)` and then mapping to load images:
    # vis_train_metadata = load_dataset("json", data_files=os.path.join(data_args.visible_dataset_dir, "train", "metadata.jsonl"))["train"]
    # def load_image_from_metadata(example):
    #     image_path = os.path.join(data_args.visible_dataset_dir, "train", "images", example["file_name"])
    #     example["image"] = Image.open(image_path)
    #     example["objects"] = example["objects"] # Assuming 'objects' is already in metadata.jsonl
    #     return example
    # vis_custom_data_train = vis_train_metadata.map(load_image_from_metadata)
    # (Repeat for val/test and for IR dataset)
    # This is a more realistic way to load your custom object detection data.
    # The original script's use of `imagefolder` for object detection data with complex metadata is unusual
    # unless there's a custom loader or specific directory structure not shown.

    logging.info(f"Loading IR dataset from: {data_args.ir_dataset_dir}")
    ir_custom_data = load_dataset("imagefolder", data_dir=data_args.ir_dataset_dir, drop_labels=True)


    # The following assumes vis_custom_data and ir_custom_data are DatasetDicts
    # with "train", "validation", "test" splits and that they have compatible row counts after shuffling.
    # Sorting by image_id and shuffling was in the original. Replicating shuffle.
    # This requires an 'image_id' column, which `imagefolder` does not provide by default.
    # You must add 'image_id' to your datasets after loading, e.g., from filenames.
    # For example:
    def add_image_id_from_filename(example, idx, prefix): # Added idx for unique IDs
        filename = example['image'].filename # Path to image
        base_name = os.path.splitext(os.path.basename(filename))[0]
        example[f'{prefix}_image_id'] = base_name # Or a more robust ID
        # Placeholder for objects, this needs to come from your metadata.jsonl
        # This is a critical part that needs to be correctly implemented based on how S4_...
        # script saves data and how you intend to load it.
        # The original script's `examples["ir_objects"]` must be populated.
        if f'{prefix}_objects' not in example:
             example[f'{prefix}_objects'] = {'bbox': [], 'category': [], 'area': []} # Dummy
        return example


    processed_splits = {}
    for split in ["train", "validation", "test"]:
        if split not in vis_custom_data or split not in ir_custom_data:
            logging.warning(f"Split '{split}' not found in both datasets. Skipping this split.")
            continue

        vis_split_data = vis_custom_data[split].map(add_image_id_from_filename, with_indices=True, fn_kwargs={"prefix": "vis"})
        ir_split_data = ir_custom_data[split].map(add_image_id_from_filename, with_indices=True, fn_kwargs={"prefix": "ir"})
        
        # The original script sorted by 'image_id'. If 'image_id' is from filename, it might be string sort.
        # Ensure that after adding image_id, the datasets can be meaningfully sorted and concatenated row-wise.
        # This assumes that the Nth image in vis_split_data corresponds to Nth image in ir_split_data AFTER shuffling both with the same seed.
        vis_split_data = vis_split_data.shuffle(seed=42) # Use a fixed seed for reproducibility
        ir_split_data = ir_split_data.shuffle(seed=42) # Must use the same seed!

        # Rename columns
        vis_renamed = vis_split_data.rename_columns({col: f"vis_{col}" for col in vis_split_data.column_names if not col.startswith("vis_")})
        ir_renamed = ir_split_data.rename_columns({col: f"ir_{col}" for col in ir_split_data.column_names if not col.startswith("ir_")})
        
        # Ensure same number of rows before concatenating
        if len(vis_renamed) != len(ir_renamed):
            logging.warning(f"Row count mismatch in '{split}' split: Visible ({len(vis_renamed)}) vs IR ({len(ir_renamed)}). Truncating to shorter length.")
            min_len = min(len(vis_renamed), len(ir_renamed))
            vis_renamed = vis_renamed.select(range(min_len))
            ir_renamed = ir_renamed.select(range(min_len))

        combined_split = concatenate_datasets([vis_renamed, ir_renamed], axis=1)
        processed_splits[split] = combined_split

    if not processed_splits:
        raise ValueError("No dataset splits could be processed. Check data paths and split names.")
        
    combined_dataset = DatasetDict(processed_splits)

    # Setup augmentations
    train_alb_transform = A.Compose([
        A.Compose([
            A.SmallestMaxSize(max_size=data_args.image_size, p=1.0),
            A.RandomSizedBBoxSafeCrop(height=data_args.image_size, width=data_args.image_size, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.Blur(blur_limit=7, p=0.5), A.MotionBlur(blur_limit=7, p=0.5),
            A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
        ], p=0.1),
        A.Perspective(p=0.1), A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.1),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1.0, min_visibility=0.3))

    eval_alb_transform = A.Compose(
        [A.NoOp()], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True)
    )

    # Apply transforms to datasets
    transform_kwargs_base = {
        "image_processor": image_processor,
        "all_conditions_data": all_conditions,
        "indices_to_sample_conditions": indices_to_sample_conditions,
    }
    
    combined_dataset["train"] = combined_dataset["train"].with_transform(
        partial(augment_and_transform_batch_multimodal, transform=train_alb_transform, current_split_name="train", **transform_kwargs_base)
    )
    if "validation" in combined_dataset:
        combined_dataset["validation"] = combined_dataset["validation"].with_transform(
            partial(augment_and_transform_batch_multimodal, transform=eval_alb_transform, current_split_name="validation", **transform_kwargs_base)
        )
    if "test" in combined_dataset:
        combined_dataset["test"] = combined_dataset["test"].with_transform(
            partial(augment_and_transform_batch_multimodal, transform=eval_alb_transform, current_split_name="test", **transform_kwargs_base)
        )
        
    return combined_dataset

# --- Main Function ---
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        model_dir_1=model_args.model_dir_1,
        model_dir_2=model_args.model_dir_2,
        config=model_config, # Pass the updated config
        ensemble_method=model_args.ensemble_method,
        n_conditions=n_conditions_eff # Pass the effective number of conditions
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
    # CRITICAL NOTE: The following `prepare_datasets` call assumes that `load_dataset("imagefolder", ...)`
    # correctly loads your object detection annotations (`image_id`, `image`, `objects`).
    # This is often NOT the case by default. You likely need to replace the `imagefolder` loading
    # with loading from your `metadata.jsonl` files and then mapping to load images.
    # The current placeholder `add_image_id_from_filename` and dummy objects in `prepare_datasets`
    # MUST BE REPLACED with your actual data loading logic for object detection annotations.
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

    trainer = Trainer(
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
            wandb.init(project="ECAI-ATR", entity="EDCR", name=training_args.run_name, config=vars(training_args)) # Add model_args, data_args to config if desired
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Original placement
    main()