import os
import pandas as pd
import argparse
from config_utils import SCENARIOS, CLASSES_FILTER, TGTTYPE_TO_ID, read_json_file

def get_filtered_unique_tags(video_dir, scenarios_filter, classes_filter_list, file_extension=".avi"):
    """
    Scans a directory for video files, filters them based on scenario and class in their filenames,
    and returns a list of unique tags.
    A tag is typically the filename without extension (e.g., 'cegr02003_0001').
    """
    unique_tags = set()
    if not os.path.isdir(video_dir):
        print(f"Warning: Video directory not found: {video_dir}")
        return []

    for filename in os.listdir(video_dir):
        if filename.endswith(file_extension):
            tag = filename.split('.')[0]
            try:
                parts = tag.split('_')
                if len(parts) < 2:
                    # print(f"Warning: Filename {filename} does not match expected tag format (e.g., prefixYYYY_CCCC). Skipping.")
                    continue
                
                scene_code_full = parts[0] # e.g., 'cegr02003' or 'i1co02003'
                class_code_str = parts[1].split('.')[0] # e.g., '0001' -> 1

                # Extract scene number (e.g., '02003' from 'cegr02003')
                scene_num_str = scene_code_full[-5:] # Assuming prefix is always 4 chars like 'cegr' or 'i1co'
                
                cls = int(class_code_str)

                if scene_num_str in scenarios_filter and cls in classes_filter_list:
                    unique_tags.add(tag)
            except ValueError:
                print(f"Warning: Could not parse class from filename {filename}. Skipping.")
            except IndexError:
                print(f"Warning: Filename {filename} format unexpected. Skipping.")
    return sorted(list(unique_tags))

def synchronize_ir_rgb_tags(ir_tags_raw, rgb_tags_raw):
    """
    Synchronizes IR tags with RGB tags.
    An IR tag is kept if a corresponding RGB tag exists (e.g., 'cegrXXXX_YYYY' -> 'i1coXXXX_YYYY').
    Returns a tuple: (filtered_ir_tags, corresponding_rgb_tags_for_filtered_ir)
    """
    synchronized_ir_tags = []
    corresponding_rgb_tags = []
    
    rgb_tags_set = set(rgb_tags_raw) # For efficient lookup

    for ir_tag in ir_tags_raw:
        if len(ir_tag) < 5: # Ensure tag is long enough for slicing
            print(f"Warning: IR tag '{ir_tag}' is too short. Skipping synchronization for this tag.")
            continue
        
        # Construct potential corresponding RGB tag
        # Assumes IR tags start with something like 'cegr' and RGB tags with 'i1co'
        # and the rest of the tag (scenario_class) is identical.
        rgb_equivalent_tag = 'i1co' + ir_tag[4:] 

        if rgb_equivalent_tag in rgb_tags_set:
            synchronized_ir_tags.append(ir_tag)
            corresponding_rgb_tags.append(rgb_equivalent_tag)
        # else:
            # print(f"Debug: IR tag {ir_tag} has no corresponding RGB tag {rgb_equivalent_tag}. It will be excluded.")

    return synchronized_ir_tags, corresponding_rgb_tags

def extract_dataframe_from_agt_json(tags_to_process, agt_json_root_dir, tgttype_to_id_map, sensor_prefix=""):
    """
    Processes AGT-JSON files for a list of tags and creates a Pandas DataFrame.
    """
    processed_data_list = []
    valid_target_types = set(tgttype_to_id_map.keys())

    for tag_identifier in tags_to_process:
        json_file_path = os.path.join(agt_json_root_dir, f"{tag_identifier}.json")
        
        agt_data_content = read_json_file(json_file_path)
        if agt_data_content is None:
            print(f"Warning: Could not read or parse JSON for tag {tag_identifier}. Skipping.")
            continue

        try:
            # Navigate through the expected JSON structure
            tgt_updates = agt_data_content.get('Agt', {}).get('TgtSect', {}).get('TgtUpd', [])
            if not isinstance(tgt_updates, list):
                print(f"Warning: 'TgtUpd' is not a list or not found for tag {tag_identifier} in {json_file_path}. Skipping.")
                continue

            for frame_index, frame_info in enumerate(tgt_updates):
                if not isinstance(frame_info, dict):
                    # print(f"Warning: Frame info at index {frame_index} for tag {tag_identifier} is not a dictionary. Skipping.")
                    continue

                target_details = frame_info.get('Tgt')
                frame_time_raw = frame_info.get('Time')

                if target_details is None or frame_time_raw is None:
                    # print(f"Warning: 'Tgt' or 'Time' data missing in frame {frame_index} for tag {tag_identifier}. Skipping frame.")
                    continue
                
                targets_in_frame = []
                if isinstance(target_details, list):
                    targets_in_frame.extend(target_details)
                elif isinstance(target_details, dict):
                    targets_in_frame.append(target_details)
                else:
                    # print(f"Warning: 'Tgt' data has unexpected type in frame {frame_index} for tag {tag_identifier}. Skipping frame.")
                    continue

                for tgt_data in targets_in_frame:
                    if not isinstance(tgt_data, dict):
                        # print(f"Warning: Target data item is not a dictionary in frame {frame_index} for tag {tag_identifier}. Skipping target.")
                        continue

                    tgt_type = tgt_data.get('TgtType')
                    if tgt_type not in valid_target_types:
                        # print(f"Info: Target type '{tgt_type}' in tag {tag_identifier}, frame {frame_index+1} is not in TGTTYPE_TO_ID. Skipping target.")
                        continue

                    try:
                        tgt_dict = {
                            'frame': frame_index + 1, # 1-based frame index
                            'tag': tag_identifier,
                            'range': float(tgt_data.get('Range', [0.0])[0]) if isinstance(tgt_data.get('Range'), list) and tgt_data.get('Range') else float(tgt_data.get('Range', 0.0)),
                            'center_pixel_x': float(tgt_data.get('PixLoc', [0.0, 0.0])[0]) if isinstance(tgt_data.get('PixLoc'), list) and len(tgt_data.get('PixLoc')) > 0 else 0.0,
                            'center_pixel_y': float(tgt_data.get('PixLoc', [0.0, 0.0])[1]) if isinstance(tgt_data.get('PixLoc'), list) and len(tgt_data.get('PixLoc')) > 1 else 0.0,
                            'aspect_angle': float(tgt_data.get('Aspect', [0.0])[0]) if isinstance(tgt_data.get('Aspect'), list) and tgt_data.get('Aspect') else float(tgt_data.get('Aspect', 0.0)),
                            'tgt_type': tgt_type,
                            'tgt_id': tgttype_to_id_map[tgt_type],
                            'time_raw': frame_time_raw # Keep raw time for later conversion
                        }
                        processed_data_list.append(tgt_dict)
                    except (TypeError, ValueError, IndexError) as e:
                        print(f"Error processing target data for tag {tag_identifier}, frame {frame_index+1}: {e}. Data: {tgt_data}. Skipping target.")

        except KeyError as e:
            print(f"Warning: Missing key {e} in JSON structure for tag {tag_identifier} in {json_file_path}. Skipping tag.")
        except Exception as e:
            print(f"An unexpected error occurred while processing tag {tag_identifier}: {e}. Skipping tag.")
            
    if not processed_data_list:
        print(f"Warning: No data processed for sensor prefix '{sensor_prefix}'. Returning empty DataFrame.")
        return pd.DataFrame()
        
    return pd.DataFrame(processed_data_list)

def main():
    parser = argparse.ArgumentParser(description="Prepare initial IR and RGB metadata DataFrames from AGT JSONs and AVI metadata.")
    parser.add_argument("--rgb_video_dir", required=True, help="Path to RGB AVI video files directory (e.g., .../i1co/arf-video).")
    parser.add_argument("--ir_video_dir", required=True, help="Path to IR AVI video files directory (e.g., .../cegr/arf-video).")
    parser.add_argument("--rgb_agt_json_dir", required=True, help="Path to RGB AGT-JSON files directory (e.g., .../i1co/agt-json).")
    parser.add_argument("--ir_agt_json_dir", required=True, help="Path to IR AGT-JSON files directory (e.g., .../cegr/agt-json).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV DataFrames.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Step 1: Filtering unique tags from AVI filenames...")
    raw_rgb_tags = get_filtered_unique_tags(args.rgb_video_dir, SCENARIOS, CLASSES_FILTER)
    raw_ir_tags = get_filtered_unique_tags(args.ir_video_dir, SCENARIOS, CLASSES_FILTER)
    print(f"Found {len(raw_rgb_tags)} raw RGB tags and {len(raw_ir_tags)} raw IR tags.")

    print("\nStep 2: Synchronizing IR and RGB tags...")
    synchronized_ir_tags, corresponding_rgb_tags = synchronize_ir_rgb_tags(raw_ir_tags, raw_rgb_tags)
    print(f"Found {len(synchronized_ir_tags)} IR tags with corresponding RGB tags.")
    if not synchronized_ir_tags:
        print("No synchronized tags found. Exiting.")
        return

    print("\nStep 3: Processing AGT-JSON files for RGB data...")
    i1co_df = extract_dataframe_from_agt_json(corresponding_rgb_tags, args.rgb_agt_json_dir, TGTTYPE_TO_ID, sensor_prefix="i1co")
    if not i1co_df.empty:
        output_path_rgb = os.path.join(args.output_dir, "i1co_initial_metadata.csv")
        i1co_df.to_csv(output_path_rgb, index=False)
        print(f"RGB DataFrame saved to {output_path_rgb} with {len(i1co_df)} records.")
    else:
        print("No RGB data processed.")

    print("\nStep 4: Processing AGT-JSON files for IR data...")
    cegr_df = extract_dataframe_from_agt_json(synchronized_ir_tags, args.ir_agt_json_dir, TGTTYPE_TO_ID, sensor_prefix="cegr")
    if not cegr_df.empty:
        output_path_ir = os.path.join(args.output_dir, "cegr_initial_metadata.csv")
        cegr_df.to_csv(output_path_ir, index=False)
        print(f"IR DataFrame saved to {output_path_ir} with {len(cegr_df)} records.")
    else:
        print("No IR data processed.")
        
    print("\nInitial DataFrame preparation complete.")

if __name__ == "__main__":
    main()
