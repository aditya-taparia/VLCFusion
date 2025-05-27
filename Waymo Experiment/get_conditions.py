import os
import json
import base64
import time
import re
from io import BytesIO
from PIL import Image
import numpy as np
import torch 
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import mmengine # For loading Waymo .pkl files
from typing import List, Dict, Any, Optional, Union

# --- Configuration ---
DEFAULT_OPENAI_MODEL = "gpt-4o" # Or "gpt-4o-2024-11-20"
DEFAULT_MAX_API_RETRIES = 3
DEFAULT_API_REQUEST_DELAY_SECONDS = 0.75
DEFAULT_MAX_WORKERS = 10 # Adjust as needed

# --- Helper Functions ---

def load_questions(file_path: str) -> List[str]:
    """Loads a list of questions from a JSON file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            questions = json.load(f)
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError("Questions file should contain a JSON list of strings.")
        return questions
    except FileNotFoundError:
        print(f"Error: Questions file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from questions file: {file_path}")
        raise
    except ValueError as e:
        print(f"Error: Invalid format in questions file {file_path}: {e}")
        raise

def format_questions_for_prompt(questions_list: List[str]) -> str:
    """Formats the list of questions into a numbered string for the prompt."""
    return "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions_list)])

def preprocess_image_to_pil(image_input: Union[np.ndarray, torch.Tensor, Image.Image, str]) -> Image.Image:
    """
    Converts various image input types (numpy array, PyTorch tensor, PIL Image, or file path string)
    to a PIL Image object in RGB format.
    """
    if isinstance(image_input, str): # If it's a file path
        try:
            image_input = Image.open(image_input)
        except FileNotFoundError:
            raise ValueError(f"Image file not found at path: {image_input}")
        except Exception as e:
            raise ValueError(f"Failed to open image from path {image_input}: {e}")

    if isinstance(image_input, Image.Image):
        return image_input.convert('RGB')

    if hasattr(image_input, "detach"): # Check for PyTorch tensor
        image_input = image_input.detach().cpu().numpy()

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[0] in [1, 3]: # Channel-first (C, H, W)
            image_input = np.transpose(image_input, (1, 2, 0))
        if image_input.ndim == 2: # Grayscale (H, W)
            image_input = np.stack((image_input,)*3, axis=-1) # Convert to (H, W, 3)
        if image_input.dtype in [np.float32, np.float64]:
            if image_input.min() < 0 or image_input.max() > 1: # If not in 0-1 range
                image_input = (image_input - image_input.min()) / (image_input.max() - image_input.min() + 1e-6)
            image_input = (image_input * 255).astype(np.uint8)
        if image_input.shape[-1] == 1: # Grayscale that was [H,W,1]
             image_input = np.concatenate([image_input]*3, axis=-1)
        try:
            return Image.fromarray(image_input).convert('RGB')
        except Exception as e:
            raise ValueError(
                "Failed to convert numpy array to PIL image. Ensure array has shape (H,W,C) and dtype uint8."
            ) from e
    
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    """Encodes a PIL Image object to a base64 string."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def save_results_to_jsonl(results_list: List[Dict[str, Any]], output_path: str):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results_list:
                f.write(json.dumps(item) + '\n')
        print(f"Successfully saved results to {output_path}")
    except IOError as e:
        print(f"Error: Could not write results to file {output_path}: {e}")

def load_waymo_data_list(annotation_pkl_path: str) -> List[Dict[str, Any]]:
    """Loads the 'data_list' from a Waymo annotation .pkl file using mmengine."""
    try:
        data = mmengine.load(annotation_pkl_path)
        if isinstance(data, dict) and 'data_list' in data and isinstance(data['data_list'], list):
            return data['data_list']
        else:
            raise ValueError("Loaded .pkl file does not contain a 'data_list' key with a list value, or is not a dictionary.")
    except FileNotFoundError:
        print(f"Error: Waymo annotation .pkl file not found at {annotation_pkl_path}")
        raise
    except Exception as e:
        print(f"Error loading Waymo annotation .pkl file {annotation_pkl_path}: {e}")
        raise

# --- Core Logic: OpenAI API Interaction (largely same as previous refactoring) ---

def fetch_image_conditions_from_api(
    client: OpenAI,
    base64_image: str,
    formatted_questions_prompt: str,
    num_questions: int,
    model: str,
    max_retries: int,
    api_request_delay: float
) -> Optional[List[bool]]:
    """
    Fetches conditions for an image from the OpenAI API.
    Returns a list of booleans corresponding to the questions, or None if an error occurs after retries.
    """
    message_text = (
        "Answer the following questions based on the given image:\n"
        "## Questions:\n"
        f"{formatted_questions_prompt}\n\n"
        f"IMPORTANT: Your answer must be a JSON object with exactly {num_questions} keys. "
        f"The keys should be the numbers from 1 to {num_questions} (as strings), and each value must be a boolean (True or False), "
        "one for each question, and nothing else. The image is provided after these questions."
    )
    
    system_prompt = (
        "You are a highly specialized assistant that provides concise answers to specific questions about images. "
        "For each question, respond with either True or False only. "
        f"Provide your answer as a JSON object with keys corresponding to the question numbers (from 1 to {num_questions}). "
        "Do not provide additional context or descriptions."
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                # response_format={"type": "json_object"} # Consider enabling if your OpenAI model/version supports it well
            )

            if not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
                print(f"Warning: Empty response from API for an image (Attempt {attempt+1}/{max_retries}). Retrying...")
                if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                continue

            response_text = response.choices[0].message.content.strip()
            response_text = re.sub(r"```(?:json)?", "", response_text).strip() # Remove markdown fences

            try:
                conditions_dict = json.loads(response_text)
            except json.JSONDecodeError:
                print(f"Warning: API response was not valid JSON (Attempt {attempt+1}). Trying fallback parse. Response: '{response_text[:100]}...'")
                try: # Fallback for list-like string e.g., "[True, False, True]"
                    list_str_content = response_text.strip("[] ")
                    if not list_str_content: raise ValueError("Empty list string after stripping brackets.")
                    parsed_bools = [item.strip().lower() == "true" for item in list_str_content.split(",")]
                    if len(parsed_bools) == num_questions:
                        conditions_dict = {str(i+1): val for i, val in enumerate(parsed_bools)}
                    else:
                        raise ValueError(f"Fallback parsed list length ({len(parsed_bools)}) != num_questions ({num_questions})")
                except Exception as e_fallback:
                    print(f"Error during fallback parsing (Attempt {attempt+1}): {e_fallback}. Retrying if possible.")
                    if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                    continue

            if not isinstance(conditions_dict, dict) or len(conditions_dict) != num_questions:
                print(f"Warning: Response dict structure mismatch (got {len(conditions_dict) if isinstance(conditions_dict, dict) else type(conditions_dict)} keys, expected {num_questions}) (Attempt {attempt+1}). Retrying...")
                if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                continue
            
            parsed_conditions = [None] * num_questions
            valid_response = True
            for i in range(num_questions):
                key = str(i+1)
                if key not in conditions_dict or not isinstance(conditions_dict[key], bool):
                    print(f"Warning: Invalid key '{key}' or non-boolean value in API response dict (Attempt {attempt+1}). Response: {conditions_dict}")
                    valid_response = False; break
                parsed_conditions[i] = conditions_dict[key]
            
            if valid_response: return parsed_conditions
            if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))

        except Exception as e:
            print(f"Error during API call for an image (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(api_request_delay * (attempt + 1))
            else:
                print(f"Failed to get conditions after {max_retries} attempts due to API error.")
                return None
    
    print(f"Failed to get a valid response structure for an image after {max_retries} attempts.")
    return None


# --- Per-Entry Processing Function ---

def process_waymo_data_entry(
    waymo_entry: Dict[str, Any],
    root_image_path_prefix: str,
    questions_prompt_str: str,
    num_questions_val: int,
    openai_client_obj: OpenAI,
    config_dict: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Processes a single Waymo data entry: loads image, gets conditions from API.
    Returns a dictionary with original info + conditions, or None if processing fails.
    """
    sample_idx = waymo_entry.get('sample_idx', waymo_entry.get('token')) # Use sample_idx or token as a unique ID
    if sample_idx is None:
        print(f"Warning: Skipping entry due to missing 'sample_idx' or 'token': {waymo_entry.get('lidar_points', {}).get('lidar_path', 'Unknown_entry')}")
        return None

    try:
        # Path construction for Waymo based on original script
        # Assuming 'CAM_FRONT' is always present, add error handling if not
        cam_front_info = waymo_entry.get('images', {}).get('CAM_FRONT', {})
        relative_img_path = cam_front_info.get('img_path')
        if not relative_img_path:
            print(f"Warning: Missing 'img_path' for CAM_FRONT in sample_idx {sample_idx}. Skipping.")
            return None
        
        full_img_path = os.path.join(root_image_path_prefix, relative_img_path)
        pil_image = preprocess_image_to_pil(full_img_path) # Handles path string
        base64_image = encode_pil_image_to_base64(pil_image)
        
        conditions_list = fetch_image_conditions_from_api(
            openai_client_obj,
            base64_image,
            questions_prompt_str,
            num_questions_val,
            model=config_dict.get("openai_model", DEFAULT_OPENAI_MODEL),
            max_retries=config_dict.get("max_api_retries", DEFAULT_MAX_API_RETRIES),
            api_request_delay=config_dict.get("api_request_delay_seconds", DEFAULT_API_REQUEST_DELAY_SECONDS)
        )
        
        output_entry = {
            'sample_idx': sample_idx, # Or 'token' if that's preferred
            'original_img_path': relative_img_path, # Storing relative path for reference
            'conditions': conditions_list if conditions_list else [] # Ensure empty list on failure
        }
        if conditions_list is None: # Explicit failure from API
             print(f"Warning: Failed to get conditions for sample_idx: {sample_idx}, img_path: {relative_img_path}")
        
        # Optional: Add a short delay here if making many rapid calls within the ThreadPool,
        # though the API delay in fetch_image_conditions_from_api handles retries.
        # time.sleep(config_dict.get("inter_request_delay", 0.1)) # e.g., a small base delay
        
        return output_entry

    except FileNotFoundError:
        print(f"Warning: Image file not found for sample_idx {sample_idx} at {full_img_path}. Skipping.")
    except ValueError as ve: # From preprocess_image or missing keys
        print(f"Warning: Data or image processing error for sample_idx {sample_idx} ({relative_img_path if 'relative_img_path' in locals() else 'path unknown'}): {ve}. Skipping.")
    except Exception as e:
        print(f"Warning: Unexpected error processing sample_idx {sample_idx} ({relative_img_path if 'relative_img_path' in locals() else 'path unknown'}): {e}. Skipping.")
    return None


# --- Main Orchestration ---

def main(args: argparse.Namespace):
    """Main function to orchestrate loading Waymo data, processing images for conditions, and saving results."""
    
    api_key = os.getenv("OPENAI_API_KEY", args.api_key)
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api_key.")
        return
    client = OpenAI(api_key=api_key)

    try:
        questions_list = load_questions(args.questions_file)
        if not questions_list: print("Error: No questions loaded. Exiting."); return
        formatted_questions_prompt = format_questions_for_prompt(questions_list)
        num_questions = len(questions_list)
    except Exception as e:
        print(f"Failed to load or format questions from {args.questions_file}: {e}"); return

    try:
        waymo_data_list = load_waymo_data_list(args.waymo_anno_pkl)
        if not waymo_data_list: print("Error: No data loaded from Waymo annotation file. Exiting."); return
    except Exception as e:
        print(f"Failed to load Waymo data from {args.waymo_anno_pkl}: {e}"); return

    processing_config = {
        "openai_model": args.model,
        "max_api_retries": args.max_retries,
        "api_request_delay_seconds": args.api_delay,
    }

    processed_results = []
    
    print(f"Processing {len(waymo_data_list)} Waymo data entries using up to {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_entry_id = {
            executor.submit(
                process_waymo_data_entry,
                entry,
                args.waymo_image_root_path,
                formatted_questions_prompt,
                num_questions,
                client,
                processing_config
            ): entry.get('sample_idx', entry.get('token', 'unknown_id'))
            for entry in waymo_data_list
        }

        for future in tqdm(as_completed(future_to_entry_id), total=len(waymo_data_list), desc="Processing Waymo entries"):
            entry_id_for_progress = future_to_entry_id[future]
            try:
                result = future.result()
                if result: # Only append if processing was successful and returned data
                    processed_results.append(result)
            except Exception as e: # Catch errors from the future.result() call itself
                print(f"Error processing future for entry_id '{entry_id_for_progress}': {e}")
    
    # Post-processing check (from original script)
    if processed_results:
        print("\nCondition length check post-processing:")
        for data_item in processed_results:
            if 'conditions' in data_item and isinstance(data_item['conditions'], list):
                if len(data_item['conditions']) != num_questions:
                    print(f"  Sample {data_item.get('sample_idx', 'Unknown')} has {len(data_item['conditions'])} conditions instead of {num_questions}.")
            elif 'conditions' not in data_item:
                 print(f"  Sample {data_item.get('sample_idx', 'Unknown')} is missing 'conditions' key.")


        save_results_to_jsonl(processed_results, args.output_jsonl_file)
    else:
        print("No results were successfully processed or collected.")

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Waymo images to get VLM conditions using OpenAI API.")
    parser.add_argument(
        "waymo_image_root_path",
        type=str,
        help="Root directory where Waymo images (e.g., 'training/image_0/') are located."
    )
    parser.add_argument(
        "waymo_anno_pkl",
        type=str,
        help="Path to the Waymo annotation .pkl file (e.g., 'waymo_infos_val.pkl')."
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="config/refined_conditions_questions.json", # Example default path
        help="Path to the JSON file containing the list of questions."
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        default="output/waymo_vlm_conditions.jsonl", # Example default path
        help="Path to save the output JSONL file with conditions."
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_OPENAI_MODEL})."
    )
    parser.add_argument(
        "--max_retries", type=int, default=DEFAULT_MAX_API_RETRIES,
        help=f"Max retries for API calls (default: {DEFAULT_MAX_API_RETRIES})."
    )
    parser.add_argument(
        "--api_delay", type=float, default=DEFAULT_API_REQUEST_DELAY_SECONDS,
        help=f"Base delay (s) between API retries (default: {DEFAULT_API_REQUEST_DELAY_SECONDS})."
    )
    parser.add_argument(
        "--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Max worker threads for concurrent processing (default: {DEFAULT_MAX_WORKERS})."
    )

    parsed_args = parser.parse_args()
    main(parsed_args)