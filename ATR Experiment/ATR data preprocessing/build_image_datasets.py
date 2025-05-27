import os
import pandas as pd
import numpy as np
import cv2
import json
import argparse
from sklearn.model_selection import train_test_split # For splitting
from config_utils import (
    TGTID_TO_CATEGORIES, TGTTYPE_TO_ID, 
    read_json_file, convert_to_serializable_for_json, VALID_TGTTYPES_FOR_CATEGORIES
)

def create_dataset_directories(base_save_path, sensor_name):
    """Creates train, test, and val directories for images."""
    sensor_path = os.path.join(base_save_path, sensor_name)
    splits = ["train", "test", "val"]
    paths = {}
    for split in splits:
        img_path = os.path.join(sensor_path, split, "images")
        os.makedirs(img_path, exist_ok=True)
        paths[split] = {
            "images": img_path,
            "metadata_file": os.path.join(sensor_path, split, "metadata.jsonl")
        }
    return paths

def process_ir_split(data_df, split_name, video_base_path_ir, agt_json_base_path_ir, metric_base_path, output_paths_ir):
    """Processes a data split (train, val, or test) for the IR dataset."""
    metadata_list_ir = []
    image_id_counter = 0
    
    print(f"  Processing IR {split_name} split ({len(data_df)} items)...")

    for idx, row in data_df.iterrows():
        ir_tag = row['tag_ir']
        ir_frame_num = int(row['frame_ir'])
        
        video_file_path = os.path.join(video_base_path_ir, f"{ir_tag}.avi")
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"    Warning: Could not open IR video {video_file_path}. Skipping frame {ir_frame_num}.")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, ir_frame_num - 1)
        ret, frame_image = cap.read()
        cap.release()

        if not ret:
            print(f"    Warning: Could not read IR frame {ir_frame_num} from {video_file_path}. Skipping.")
            continue

        image_filename = f"{ir_tag}_{ir_frame_num}.png"
        cv2.imwrite(os.path.join(output_paths_ir[split_name]['images'], image_filename), frame_image)

        # --- Metadata Extraction ---
        obj_bboxes = []
        obj_categories_name = []
        obj_categories_id = []
        obj_areas = []
        obj_ranges = []
        obj_aspect_angles = []
        obj_center_pixels_json = [] # From AGT JSON

        # 1. AGT JSON for ranges, aspect angles, target types, center pixels
        json_agt_path = os.path.join(agt_json_base_path_ir, f"{ir_tag}.json")
        agt_content = read_json_file(json_agt_path)
        if not agt_content:
            print(f"    Warning: Could not read AGT JSON {json_agt_path} for IR. Skipping metadata for this frame.")
            continue
        
        try:
            frame_specific_data = agt_content['Agt']['TgtSect']['TgtUpd'][ir_frame_num - 1]['Tgt']
            targets_in_agt = []
            if isinstance(frame_specific_data, list):
                targets_in_agt.extend(frame_specific_data)
            elif isinstance(frame_specific_data, dict):
                 targets_in_agt.append(frame_specific_data)

            for tgt_entry in targets_in_agt:
                tgt_type = tgt_entry.get('TgtType')
                if tgt_type not in VALID_TGTTYPES_FOR_CATEGORIES: # Use only valid types for categorization
                    continue 

                obj_ranges.append(float(tgt_entry.get('Range', [0.0])[0]))
                obj_aspect_angles.append(float(tgt_entry.get('Aspect', [0.0])[0]))
                obj_categories_name.append(tgt_type)
                obj_categories_id.append(TGTID_TO_CATEGORIES[TGTTYPE_TO_ID[tgt_type]])
                obj_center_pixels_json.append(tgt_entry.get('PixLoc', [0.0, 0.0]))


            # 2. Metric file for upper-left of bounding box
            metric_file_path = os.path.join(metric_base_path, f"{ir_tag}.bbox_met")
            upper_left_pixels_metric = []
            if os.path.exists(metric_file_path):
                metric_df = pd.read_csv(metric_file_path, header=None, 
                                        names=['Site', 'Na', 'Base', 'Sensor', 'Scenario', 
                                               'Frame', 'Ply_id', 'SNR', 'NULL', 'Upper_left_x', 
                                               'Upper_left_y', 'Mean_tgt', 'Std_tgt', 'POT', 
                                               'Eff_pot', 'Mean_bkg', 'Std_bkg', 'POB'])
                frame_metric_data = metric_df[metric_df['Frame'] == ir_frame_num]
                
                # Match metric entries to AGT entries - assume order or use Ply_id if available and reliable
                # For simplicity, if counts match, assume direct correspondence.
                # This part might need refinement if there are multiple targets and order isn't guaranteed.
                if len(frame_metric_data) == len(obj_center_pixels_json):
                    for _, metric_row in frame_metric_data.iterrows():
                        upper_left_pixels_metric.append([metric_row['Upper_left_x'], metric_row['Upper_left_y']])
                elif not frame_metric_data.empty and len(obj_center_pixels_json) > 0 : # If counts don't match but some data exists
                     print(f"    Warning: Mismatch in target count between AGT JSON ({len(obj_center_pixels_json)}) and Metric file ({len(frame_metric_data)}) for {ir_tag} frame {ir_frame_num}. Using first available metric for all AGT targets or skipping bbox.")
                     # Fallback: use first metric for all, or handle more sophisticatedly
                     if len(frame_metric_data) > 0:
                        first_metric_ul = [frame_metric_data.iloc[0]['Upper_left_x'], frame_metric_data.iloc[0]['Upper_left_y']]
                        upper_left_pixels_metric = [first_metric_ul] * len(obj_center_pixels_json)


            # 3. Calculate BBoxes and Areas
            if len(upper_left_pixels_metric) == len(obj_center_pixels_json) and len(obj_center_pixels_json)>0:
                for i in range(len(obj_center_pixels_json)):
                    ul_x, ul_y = upper_left_pixels_metric[i]
                    c_x, c_y = obj_center_pixels_json[i]
                    
                    # Width/Height calculation from your IR dataset creation code
                    width = abs(c_x - ul_x) * 2
                    height = abs(c_y - ul_y) * 2
                    
                    obj_bboxes.append([float(ul_x), float(ul_y), float(width), float(height)])
                    obj_areas.append(float(width * height))
            elif len(obj_center_pixels_json) > 0: # If metric file was missing or had issues, bboxes might be empty
                 print(f"    Warning: Could not form bounding boxes for all targets in {ir_tag} frame {ir_frame_num} due to metric data issues. Some bboxes may be missing.")
                 # Fill with placeholder if necessary, or ensure downstream handles missing bboxes
                 obj_bboxes = [[0,0,0,0]] * len(obj_center_pixels_json)
                 obj_areas = [0] * len(obj_center_pixels_json)


            if not obj_categories_name: # If no valid targets were found
                continue

            metadata_list_ir.append({
                "image_id": image_id_counter,
                "file_name": image_filename,
                "height": frame_image.shape[0],
                "width": frame_image.shape[1],
                "objects": {
                    "bbox": obj_bboxes, # List of [x,y,width,height]
                    "category_name": obj_categories_name,
                    "category_id": obj_categories_id,
                    "area": obj_areas,
                    "range": obj_ranges,
                    "aspect_angle": obj_aspect_angles
                }
            })
            image_id_counter += 1
        except Exception as e:
            print(f"    Error processing metadata for IR frame {image_filename}: {e}. Skipping.")

    # Save metadata.jsonl for the IR split
    with open(output_paths_ir[split_name]['metadata_file'], 'w') as f:
        for item in metadata_list_ir:
            f.write(json.dumps(item, default=convert_to_serializable_for_json) + '\n')
    print(f"    IR {split_name} metadata saved with {len(metadata_list_ir)} entries.")
    return pd.DataFrame(metadata_list_ir) # Return as DF for RGB processing


def process_rgb_split(data_df, split_name, video_base_path_rgb, agt_json_base_path_rgb, ir_metadata_df_for_split, output_paths_rgb):
    """Processes a data split for the RGB dataset, using IR bboxes for interpolation."""
    metadata_list_rgb = []
    image_id_counter = 0
    
    print(f"  Processing RGB {split_name} split ({len(data_df)} items)...")

    # Create a lookup for IR metadata
    ir_meta_lookup = {}
    if not ir_metadata_df_for_split.empty:
        for _, ir_meta_row in ir_metadata_df_for_split.iterrows():
            ir_meta_lookup[ir_meta_row['file_name']] = ir_meta_row['objects']


    for idx, row in data_df.iterrows():
        rgb_tag = row['tag_rgb']
        rgb_frame_num = int(row['frame_rgb'])
        ir_tag_for_match = row['tag_ir'] # From the combined data, for finding the IR bbox
        ir_frame_for_match = int(row['frame_ir'])

        video_file_path = os.path.join(video_base_path_rgb, f"{rgb_tag}.avi")
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"    Warning: Could not open RGB video {video_file_path}. Skipping.")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_num - 1)
        ret, frame_image = cap.read()
        cap.release()

        if not ret:
            print(f"    Warning: Could not read RGB frame {rgb_frame_num} from {video_file_path}. Skipping.")
            continue

        image_filename = f"{rgb_tag}_{rgb_frame_num}.png"
        cv2.imwrite(os.path.join(output_paths_rgb[split_name]['images'], image_filename), frame_image)

        # --- Metadata Extraction ---
        obj_bboxes_rgb = []
        obj_categories_name_rgb = []
        obj_categories_id_rgb = []
        obj_areas_rgb = []
        obj_ranges_rgb = []
        obj_aspect_angles_rgb = []
        rgb_center_pixels_json = []


        # 1. AGT JSON for RGB target centers, ranges, aspect angles, types
        json_agt_path_rgb = os.path.join(agt_json_base_path_rgb, f"{rgb_tag}.json")
        agt_content_rgb = read_json_file(json_agt_path_rgb)
        if not agt_content_rgb:
            print(f"    Warning: Could not read AGT JSON {json_agt_path_rgb} for RGB. Skipping metadata for this frame.")
            continue

        try:
            frame_specific_data_rgb = agt_content_rgb['Agt']['TgtSect']['TgtUpd'][rgb_frame_num - 1]['Tgt']
            targets_in_agt_rgb = []
            if isinstance(frame_specific_data_rgb, list):
                targets_in_agt_rgb.extend(frame_specific_data_rgb)
            elif isinstance(frame_specific_data_rgb, dict):
                targets_in_agt_rgb.append(frame_specific_data_rgb)

            for tgt_entry in targets_in_agt_rgb:
                tgt_type = tgt_entry.get('TgtType')
                if tgt_type not in VALID_TGTTYPES_FOR_CATEGORIES:
                     continue
                obj_ranges_rgb.append(float(tgt_entry.get('Range', [0.0])[0]))
                obj_aspect_angles_rgb.append(float(tgt_entry.get('Aspect', [0.0])[0]))
                obj_categories_name_rgb.append(tgt_type)
                obj_categories_id_rgb.append(TGTID_TO_CATEGORIES[TGTTYPE_TO_ID[tgt_type]])
                rgb_center_pixels_json.append(tgt_entry.get('PixLoc', [0.0, 0.0])) # [cx, cy]

            # 2. BBox Interpolation using corresponding IR bbox
            ir_match_filename = f"{ir_tag_for_match}_{ir_frame_for_match}.png" # Note .png as saved by IR script
            corresponding_ir_objects = ir_meta_lookup.get(ir_match_filename)

            if corresponding_ir_objects and corresponding_ir_objects.get('bbox'):
                ir_bboxes_list = corresponding_ir_objects['bbox']
                
                # Assume same number of targets and order for simplicity
                # This is a critical assumption from your original script.
                num_rgb_targets = len(rgb_center_pixels_json)
                num_ir_bboxes = len(ir_bboxes_list)

                # Handle potential mismatch in number of detected objects
                min_targets = min(num_rgb_targets, num_ir_bboxes)
                if num_rgb_targets != num_ir_bboxes:
                    print(f"    Warning: Mismatch in target count for RGB frame {image_filename} ({num_rgb_targets}) and its IR match {ir_match_filename} ({num_ir_bboxes}). Interpolating for {min_targets} targets.")

                for i in range(min_targets):
                    ir_bbox = ir_bboxes_list[i] # [x1_ir, y1_ir, w_ir, h_ir]
                    if not (isinstance(ir_bbox, list) and len(ir_bbox) == 4):
                        print(f"    Warning: Malformed IR bbox {ir_bbox} for {ir_match_filename}. Skipping this target for RGB bbox.")
                        obj_bboxes_rgb.append([0,0,0,0]) # Placeholder
                        obj_areas_rgb.append(0)
                        continue

                    x1_ir, y1_ir, w_ir, h_ir = map(float, ir_bbox)
                    cx_ir = x1_ir + w_ir / 2
                    cy_ir = y1_ir + h_ir / 2
                    
                    cx_rgb, cy_rgb = map(float, rgb_center_pixels_json[i])
                    
                    delta_x = cx_rgb - cx_ir
                    delta_y = cy_rgb - cy_ir
                    
                    scale_factor = np.sqrt(2) # As per your original script
                    w_rgb = w_ir * scale_factor
                    h_rgb = h_ir * scale_factor
                    
                    x1_rgb = cx_rgb - w_rgb / 2 # center_x - new_width/2
                    y1_rgb = cy_rgb - h_rgb / 2 # center_y - new_height/2
                    
                    obj_bboxes_rgb.append([x1_rgb, y1_rgb, w_rgb, h_rgb])
                    obj_areas_rgb.append(w_rgb * h_rgb)
            
            elif len(rgb_center_pixels_json) > 0: # If IR bbox not found but RGB targets exist
                print(f"    Warning: No corresponding IR bbox found for {ir_match_filename} to interpolate for RGB frame {image_filename}. RGB bboxes will be missing.")
                obj_bboxes_rgb = [[0,0,0,0]] * len(rgb_center_pixels_json) # Placeholders
                obj_areas_rgb = [0] * len(rgb_center_pixels_json)


            if not obj_categories_name_rgb: # If no valid targets were found in AGT
                continue


            metadata_list_rgb.append({
                "image_id": image_id_counter,
                "file_name": image_filename,
                "height": frame_image.shape[0],
                "width": frame_image.shape[1],
                "objects": {
                    "bbox": obj_bboxes_rgb,
                    "category_name": obj_categories_name_rgb,
                    "category_id": obj_categories_id_rgb,
                    "area": obj_areas_rgb,
                    "range": obj_ranges_rgb,
                    "aspect_angle": obj_aspect_angles_rgb
                }
            })
            image_id_counter += 1
        except Exception as e:
            print(f"    Error processing metadata for RGB frame {image_filename}: {e}. Skipping.")

    # Save metadata.jsonl for the RGB split
    with open(output_paths_rgb[split_name]['metadata_file'], 'w') as f:
        for item in metadata_list_rgb:
            f.write(json.dumps(item, default=convert_to_serializable_for_json) + '\n')
    print(f"    RGB {split_name} metadata saved with {len(metadata_list_rgb)} entries.")


def main():
    parser = argparse.ArgumentParser(description="Build final IR and RGB image datasets with train/val/test splits.")
    parser.add_argument("--combined_csv", required=True, help="Path to the combined and synchronized data CSV.")
    parser.add_argument("--base_output_dir", required=True, help="Base directory to save the 'mwir_dataset' and 'visible_dataset'.")
    
    parser.add_argument("--ir_video_dir", required=True, help="Path to IR AVI video files (e.g., .../cegr/arf-video).")
    parser.add_argument("--rgb_video_dir", required=True, help="Path to RGB AVI video files (e.g., .../i1co/arf-video).")
    parser.add_argument("--ir_agt_json_dir", required=True, help="Path to IR AGT-JSON files (e.g., .../cegr/agt-json).")
    parser.add_argument("--rgb_agt_json_dir", required=True, help="Path to RGB AGT-JSON files (e.g., .../i1co/agt-json).")
    parser.add_argument("--metric_dir", required=True, help="Path to Metric files directory (containing .bbox_met).")

    parser.add_argument("--train_ratio", type=float, default=0.6, help="Proportion of data for training.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Proportion of data for validation.")
    # Test ratio will be 1 - train_ratio - val_ratio
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducible splits.")
    
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and (args.train_ratio + args.val_ratio) < 1):
        print("Error: Train and Val ratios must be between 0 and 1, and their sum must be less than 1.")
        return

    print("Loading combined synchronized data...")
    try:
        combined_df = pd.read_csv(args.combined_csv)
    except FileNotFoundError:
        print(f"Error: Combined CSV file not found at {args.combined_csv}")
        return
    
    # Original script does: filterd_data = data.drop_duplicates(subset=['tag_ir','tag_rgb','frame_rgb'], keep='last')
    # This was already done in S3_combine_and_synchronize.py with keep='first' after sorting.
    # If it needs to be 'last', it should be consistently applied. Assuming 'first' after sort is okay.
    print(f"Loaded {len(combined_df)} records from combined data.")
    if combined_df.empty:
        print("Combined data is empty. Cannot proceed.")
        return

    # Stratified split based on unique IR tags to ensure diversity of scenarios/classes per tag
    unique_ir_tags_for_split = combined_df['tag_ir'].unique()
    
    if len(unique_ir_tags_for_split) < 2 : # Need at least 2 groups for train_test_split to be meaningful for stratification
        print("Warning: Very few unique IR tags for stratified split. Consider a different strategy or more data.")
        # Fallback to non-stratified if only one tag, or handle as error
        if len(unique_ir_tags_for_split) == 1:
             train_tags, temp_tags = [unique_ir_tags_for_split[0]], [] # All to train if only one tag
        else: # No tags
            print("Error: No unique IR tags found for splitting. Exiting.")
            return
    else:
        train_tags, temp_tags = train_test_split(
            unique_ir_tags_for_split,
            train_size=args.train_ratio,
            random_state=args.random_state,
            # stratify=unique_ir_tags_for_split # Stratify by the tags themselves if that's the intention, or a derivative like class
        )
        if not temp_tags.size > 0: # If temp_tags is empty
             val_tags, test_tags = [], []
             print("Warning: Not enough data for validation and test splits after training split.")
        else:
            relative_val_ratio = args.val_ratio / (1.0 - args.train_ratio) # Adjust val_ratio for the remainder
            if len(temp_tags) < 2 or relative_val_ratio >= 1.0 : # Need at least 2 groups for another split
                val_tags, test_tags = temp_tags, [] # Assign all remaining to val if not enough for further split or ratio is too high
                if relative_val_ratio >=1.0 and len(temp_tags) > 0 :
                     print(f"Warning: Relative validation ratio ({relative_val_ratio:.2f}) is too high for remaining data. Assigning all to validation.")
                elif len(temp_tags) < 2 and len(temp_tags) > 0:
                     print("Warning: Not enough unique tags in temporary set for further validation/test split. Assigning all to validation.")

            else:
                val_tags, test_tags = train_test_split(
                    temp_tags,
                    train_size=relative_val_ratio,
                    random_state=args.random_state,
                )

    train_df = combined_df[combined_df['tag_ir'].isin(train_tags)]
    val_df = combined_df[combined_df['tag_ir'].isin(val_tags)]
    test_df = combined_df[combined_df['tag_ir'].isin(test_tags)]

    print(f"\nData split into: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)}) items.")

    # --- Create Dataset Directories ---
    output_paths_ir = create_dataset_directories(args.base_output_dir, "mwir_dataset_final")
    output_paths_rgb = create_dataset_directories(args.base_output_dir, "visible_dataset_final")

    # --- Process IR Dataset First (RGB depends on its metadata for bboxes) ---
    print("\nProcessing IR Dataset...")
    ir_meta_train_df = process_ir_split(train_df, "train", args.ir_video_dir, args.ir_agt_json_dir, args.metric_dir, output_paths_ir)
    ir_meta_val_df = process_ir_split(val_df, "val", args.ir_video_dir, args.ir_agt_json_dir, args.metric_dir, output_paths_ir)
    ir_meta_test_df = process_ir_split(test_df, "test", args.ir_video_dir, args.ir_agt_json_dir, args.metric_dir, output_paths_ir)

    # --- Process RGB Dataset ---
    print("\nProcessing RGB Dataset...")
    process_rgb_split(train_df, "train", args.rgb_video_dir, args.rgb_agt_json_dir, ir_meta_train_df, output_paths_rgb)
    process_rgb_split(val_df, "val", args.rgb_video_dir, args.rgb_agt_json_dir, ir_meta_val_df, output_paths_rgb)
    process_rgb_split(test_df, "test", args.rgb_video_dir, args.rgb_agt_json_dir, ir_meta_test_df, output_paths_rgb)

    print("\nDataset building complete.")

if __name__ == "__main__":
    main()