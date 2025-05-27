import os
import pandas as pd
import numpy as np
import argparse
from config_utils import julian_to_datetime

def load_and_prepare_train_exclusions(metadata_path):
    """Loads training metadata JSONL and creates a set of (tag, frame) for exclusion."""
    exclusions = set()
    if metadata_path and os.path.exists(metadata_path):
        try:
            df = pd.read_json(metadata_path, lines=True)
            # file_name is like 'tag_frame.jpg' or 'tag_frame.png'
            for _, row in df.iterrows():
                file_name_parts = row['file_name'].rsplit('_', 1)
                tag = file_name_parts[0]
                frame_str = file_name_parts[1].split('.')[0]
                exclusions.add((tag, int(frame_str)))
            print(f"Loaded {len(exclusions)} exclusions from {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not load or parse exclusion file {metadata_path}: {e}")
    return exclusions

def main():
    parser = argparse.ArgumentParser(description="Combine and synchronize IR and RGB DataFrames based on timestamps.")
    parser.add_argument("--ir_metadata_csv", required=True, help="Path to the IR metadata CSV (e.g., cegr_initial_metadata.csv).")
    parser.add_argument("--rgb_metadata_csv", required=True, help="Path to the RGB metadata CSV (e.g., i1co_initial_metadata.csv).")
    parser.add_argument("--ir_train_exclusion_jsonl", help="Optional path to IR training metadata.jsonl for exclusions.")
    parser.add_argument("--rgb_train_exclusion_jsonl", help="Optional path to RGB training metadata.jsonl for exclusions.")
    parser.add_argument("--output_csv", required=True, help="Path to save the combined and synchronized data CSV.")
    parser.add_argument("--time_tolerance_ms", type=int, default=100, help="Time tolerance in milliseconds for matching frames.")
    
    args = parser.parse_args()

    print("Loading data...")
    try:
        ir_df = pd.read_csv(args.ir_metadata_csv)
        rgb_df = pd.read_csv(args.rgb_metadata_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find input CSV file: {e}")
        return
        
    # Convert raw time (which is a string representation of a list) to datetime objects
    # Original time format is a list: [year, julian_day, hour, minute, second, millisecond]
    print("Converting time columns...")
    ir_df['time_raw_eval'] = ir_df['time_raw'].apply(eval) # eval to convert string list to actual list
    rgb_df['time_raw_eval'] = rgb_df['time_raw'].apply(eval)

    ir_df['datetime'] = ir_df['time_raw_eval'].apply(lambda t: julian_to_datetime(*t))
    rgb_df['datetime'] = rgb_df['time_raw_eval'].apply(lambda t: julian_to_datetime(*t))

    # Drop rows where datetime conversion failed
    ir_df.dropna(subset=['datetime'], inplace=True)
    rgb_df.dropna(subset=['datetime'], inplace=True)
    
    if ir_df.empty or rgb_df.empty:
        print("Error: One or both DataFrames are empty after time conversion. Cannot proceed.")
        return

    ir_train_exclusions = load_and_prepare_train_exclusions(args.ir_train_exclusion_jsonl)
    rgb_train_exclusions = load_and_prepare_train_exclusions(args.rgb_train_exclusion_jsonl)

    combined_data = []
    unique_ir_tags = ir_df['tag'].unique()
    
    print(f"\nProcessing {len(unique_ir_tags)} unique IR tags for synchronization...")

    for i, ir_tag_val in enumerate(unique_ir_tags):
        print(f"  Processing IR tag {i+1}/{len(unique_ir_tags)}: {ir_tag_val}...")
        
        # Corresponding RGB tag (e.g., cegrXXXX_YYYY -> i1coXXXX_YYYY)
        rgb_tag_val = 'i1co' + ir_tag_val[4:]

        current_ir_tag_df = ir_df[ir_df['tag'] == ir_tag_val].copy()
        current_rgb_tag_df = rgb_df[rgb_df['tag'] == rgb_tag_val].copy()

        if current_rgb_tag_df.empty:
            print(f"    Warning: No RGB data found for corresponding tag {rgb_tag_val}. Skipping IR tag {ir_tag_val}.")
            continue
            
        # Sort by datetime to potentially optimize search if needed, though direct comparison is fine for moderate sizes
        current_rgb_tag_df = current_rgb_tag_df.sort_values(by='datetime').reset_index(drop=True)

        for _, ir_row in current_ir_tag_df.iterrows():
            if (ir_row['tag'], ir_row['frame']) in ir_train_exclusions:
                # print(f"    Skipping excluded IR frame: {ir_row['tag']}, frame {ir_row['frame']}")
                continue

            ir_time = ir_row['datetime']
            
            # Calculate time differences with all RGB frames for the current tag
            time_diffs = (current_rgb_tag_df['datetime'] - ir_time).abs()
            
            min_diff_idx = time_diffs.idxmin()
            min_diff = time_diffs[min_diff_idx]

            if min_diff <= pd.Timedelta(milliseconds=args.time_tolerance_ms):
                rgb_match_row = current_rgb_tag_df.loc[min_diff_idx]
                
                if (rgb_match_row['tag'], rgb_match_row['frame']) in rgb_train_exclusions:
                    # print(f"    Skipping excluded RGB match: {rgb_match_row['tag']}, frame {rgb_match_row['frame']}")
                    continue

                # Create a dictionary for the combined data
                # Suffix columns with _ir and _rgb for clarity
                entry = {}
                for col in ir_row.index:
                    if col not in ['datetime', 'time_raw_eval']: # Exclude intermediate columns
                         entry[f"{col}_ir"] = ir_row[col]
                entry['datetime_ir'] = ir_row['datetime'].isoformat()


                for col in rgb_match_row.index:
                    if col not in ['datetime', 'time_raw_eval']:
                        entry[f"{col}_rgb"] = rgb_match_row[col]
                entry['datetime_rgb'] = rgb_match_row['datetime'].isoformat()
                
                entry['time_difference_ms'] = min_diff.total_seconds() * 1000
                combined_data.append(entry)

    if not combined_data:
        print("No synchronized data pairs found. Output CSV will be empty or not created.")
        return

    combined_df = pd.DataFrame(combined_data)
    
    # Drop duplicates based on the specific set of columns from your 'dataset creation code'
    # This implies that for a given IR tag, RGB tag, and RGB frame, we only want one entry.
    # The original 'combined_data.csv' was filtered using:
    # filterd_data = data.drop_duplicates(subset=['tag_ir','tag_rgb','frame_rgb'], keep='last')
    # Ensure these columns exist before dropping
    subset_cols = ['tag_ir', 'tag_rgb', 'frame_rgb']
    if all(col in combined_df.columns for col in subset_cols):
        print(f"\nInitial combined data has {len(combined_df)} rows.")
        combined_df.sort_values(by=['tag_ir', 'tag_rgb', 'frame_rgb', 'time_difference_ms'], inplace=True) # Sort to make keep='first' more deterministic
        combined_df.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
        print(f"After dropping duplicates on {subset_cols}, data has {len(combined_df)} rows.")
    else:
        print(f"Warning: One or more subset columns for drop_duplicates not found: {subset_cols}. Skipping duplicate removal.")


    combined_df.to_csv(args.output_csv, index=False)
    print(f"\nCombined and synchronized data saved to {args.output_csv} with {len(combined_df)} records.")

if __name__ == "__main__":
    main()
