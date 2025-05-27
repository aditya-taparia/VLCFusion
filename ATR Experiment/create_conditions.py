import os
import json
import base64
import time
import re
from io import BytesIO
from PIL import Image
import numpy as np
import torch # Keep if torch tensors are a possible input, otherwise remove
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional, Union

# --- Configuration ---
# Consider moving these to a separate config.py or using environment variables/CLI args more extensively.
DEFAULT_OPENAI_MODEL = "gpt-4o" # Or "gpt-4o-2024-11-20" if that's a specific version you need
DEFAULT_MAX_API_RETRIES = 3
DEFAULT_API_REQUEST_DELAY_SECONDS = 0.75 # Time to wait between retries/requests if needed
DEFAULT_MAX_WORKERS = 10 # Adjusted from 16, as very high numbers can sometimes lead to rate limits or issues

# --- Helper Functions ---

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSONL file, where each line is a JSON object."""
    data_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data_list.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line in {file_path}: {e} - Line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {file_path}")
        raise
    return data_list

def load_questions(file_path: str) -> List[str]:
    """Loads a list of questions from a JSON file."""
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            questions = json.load(f)
        if not isinstance(questions, list):
            raise ValueError("Questions file should contain a JSON list of strings.")
        return questions
    except FileNotFoundError:
        print(f"Error: Questions file not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from questions file: {file_path}")
        raise

def format_questions_for_prompt(questions_list: List[str]) -> str:
    """Formats the list of questions into a numbered string for the prompt."""
    return "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions_list)])

def preprocess_image_to_pil(image_input: Union[np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    """
    Converts various image input types (numpy array, PyTorch tensor, PIL Image)
    to a PIL Image object in RGB format.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert('RGB')

    if hasattr(image_input, "detach"): # Check for PyTorch tensor
        image_input = image_input.detach().cpu().numpy()

    if isinstance(image_input, np.ndarray):
        # Handle channel-first format (e.g., [C, H, W])
        if image_input.ndim == 3 and image_input.shape[0] in [1, 3]:
            image_input = np.transpose(image_input, (1, 2, 0))
        
        # Handle grayscale to RGB if necessary (though PIL usually handles it)
        if image_input.ndim == 2: # Grayscale
            image_input = np.stack((image_input,)*3, axis=-1)
        
        # Normalize and convert dtype if it's float (common for model outputs)
        if image_input.dtype in [np.float32, np.float64]:
            if image_input.min() < 0 or image_input.max() > 1: # Assuming it might not be 0-1
                image_input = (image_input - image_input.min()) / (image_input.max() - image_input.min() + 1e-6) # Normalize to 0-1
            image_input = (image_input * 255).astype(np.uint8)
        
        # Ensure it's 3 channels for RGB
        if image_input.shape[-1] == 1: # Grayscale that was [H,W,1]
             image_input = np.concatenate([image_input]*3, axis=-1)

        try:
            return Image.fromarray(image_input).convert('RGB')
        except Exception as e:
            raise ValueError(
                "Failed to convert numpy array to PIL image. Ensure array has correct shape (H, W, C) and dtype (uint8)."
            ) from e
    
    raise TypeError(f"Unsupported image type: {type(image_input)}")


def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    """Encodes a PIL Image object to a base64 string."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG") # Save as JPEG for encoding
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def save_conditions_data(conditions_map: Dict[Any, Any], output_path: str):
    """Saves the collected conditions data to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(conditions_map, file, indent=4)
        print(f"Successfully saved conditions to {output_path}")
    except IOError as e:
        print(f"Error: Could not write conditions to file {output_path}: {e}")

# --- Core Logic: OpenAI API Interaction ---

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
                # response_format={"type": "json_object"} # Recommended for newer models that support it
            )

            if not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
                print(f"Warning: Empty response from API (Attempt {attempt+1}/{max_retries}). Retrying...")
                if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                continue

            response_text = response.choices[0].message.content.strip()
            # Remove markdown code block fences if present
            response_text = re.sub(r"```(?:json)?", "", response_text).strip()

            try:
                conditions_dict = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback for non-JSON but potentially parsable list-like string (e.g., "[True, False, True]")
                # This is fragile and assumes a specific non-JSON format.
                print(f"Warning: API response was not valid JSON (Attempt {attempt+1}). Trying fallback parse. Response: '{response_text[:100]}...'")
                try:
                    list_str_content = response_text.strip("[] ")
                    if not list_str_content: # Empty list string
                        raise ValueError("Empty list string after stripping brackets.")
                    conditions_list_parsed = [item.strip().lower() == "true" for item in list_str_content.split(",")]
                    if len(conditions_list_parsed) == num_questions:
                        conditions_dict = {str(i+1): val for i, val in enumerate(conditions_list_parsed)}
                    else:
                        raise ValueError(f"Fallback parsed list length ({len(conditions_list_parsed)}) != num_questions ({num_questions})")
                except Exception as e_fallback:
                    print(f"Error during fallback parsing (Attempt {attempt+1}): {e_fallback}. Retrying if possible.")
                    if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                    continue # Go to next retry attempt

            if not isinstance(conditions_dict, dict) or len(conditions_dict) != num_questions:
                print(f"Warning: Response structure mismatch (got {len(conditions_dict) if isinstance(conditions_dict, dict) else type(conditions_dict)}, expected {num_questions} dict keys) (Attempt {attempt+1}). Retrying...")
                if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))
                continue
            
            # Ensure all keys are strings from "1" to "num_questions" and values are boolean
            parsed_conditions = [None] * num_questions
            valid_response = True
            for i in range(num_questions):
                key = str(i+1)
                if key not in conditions_dict or not isinstance(conditions_dict[key], bool):
                    print(f"Warning: Invalid key '{key}' or non-boolean value in API response (Attempt {attempt+1}). Response: {conditions_dict}")
                    valid_response = False
                    break
                parsed_conditions[i] = conditions_dict[key]
            
            if valid_response:
                return parsed_conditions

            if attempt < max_retries - 1: time.sleep(api_request_delay * (attempt + 1))


        except Exception as e:
            print(f"Error during API call (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(api_request_delay * (attempt + 1)) # Exponential backoff can be considered
            else:
                print(f"Failed to get conditions after {max_retries} attempts.")
                return None # Or raise error
    
    print(f"Failed to get a valid response structure after {max_retries} attempts.")
    return None


# --- Image Processing Function ---

def process_single_image_entry(
    image_entry: Dict[str, Any],
    image_base_path: str,
    questions_prompt: str,
    num_questions: int,
    openai_client: OpenAI,
    config: Dict[str, Any]
) -> Optional[tuple[Any, List[bool]]]:
    """
    Processes a single image entry: loads image, gets conditions from API.
    Returns a tuple (image_id, conditions_list) or None if processing fails.
    """
    image_id = image_entry.get("image_id")
    image_file_name = image_entry.get("file_name")

    if image_id is None or image_file_name is None:
        print(f"Warning: Skipping entry due to missing 'image_id' or 'file_name': {image_entry}")
        return None

    full_image_path = os.path.join(image_base_path, image_file_name)

    try:
        raw_image = Image.open(full_image_path)
        pil_image = preprocess_image_to_pil(raw_image) # Ensures RGB
        base64_image = encode_pil_image_to_base64(pil_image)
        
        conditions_list = fetch_image_conditions_from_api(
            openai_client,
            base64_image,
            questions_prompt,
            num_questions,
            model=config.get("openai_model", DEFAULT_OPENAI_MODEL),
            max_retries=config.get("max_api_retries", DEFAULT_MAX_API_RETRIES),
            api_request_delay=config.get("api_request_delay_seconds", DEFAULT_API_REQUEST_DELAY_SECONDS)
        )
        
        if conditions_list:
            return image_id, conditions_list
        else:
            print(f"Warning: Failed to get conditions for image_id: {image_id}, file: {image_file_name}")
            return image_id, [] # Or None, depending on how you want to handle failures downstream

    except FileNotFoundError:
        print(f"Warning: Image file not found for image_id {image_id} at {full_image_path}. Skipping.")
    except ValueError as ve: # From preprocess_image_to_pil
        print(f"Warning: Image processing error for image_id {image_id} ({full_image_path}): {ve}. Skipping.")
    except Exception as e:
        print(f"Warning: Unexpected error processing image_id {image_id} ({full_image_path}): {e}. Skipping.")
    return None


# --- Main Orchestration ---

def main(args: argparse.Namespace):
    """Main function to orchestrate loading data, processing images, and saving results."""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY", args.api_key)
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --api_key argument.")
        return
    client = OpenAI(api_key=api_key)

    # Load questions
    try:
        questions_list = load_questions(args.questions_file)
        if not questions_list:
            print("Error: No questions loaded. Exiting.")
            return
        formatted_questions = format_questions_for_prompt(questions_list)
        num_questions = len(questions_list)
    except Exception as e:
        print(f"Failed to load or format questions: {e}")
        return

    # Load image metadata
    metadata_file_full_path = os.path.join(args.parent_path, args.metadata_file_name)
    try:
        image_metadata_list = load_jsonl_data(metadata_file_full_path)
        if not image_metadata_list:
            print("Error: No image metadata loaded from. Exiting.")
            return
    except Exception as e:
        print(f"Failed to load image metadata: {e}")
        return

    # Prepare config for processing function
    processing_config = {
        "openai_model": args.model,
        "max_api_retries": args.max_retries,
        "api_request_delay_seconds": args.api_delay,
    }

    all_conditions_map = {}
    
    print(f"Processing {len(image_metadata_list)} images using up to {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_entry = {
            executor.submit(
                process_single_image_entry,
                entry,
                args.parent_path, # Base path for images
                formatted_questions,
                num_questions,
                client,
                processing_config
            ): entry.get("image_id", "unknown") # Store image_id for progress tracking
            for entry in image_metadata_list
        }

        for future in tqdm(as_completed(future_to_entry), total=len(image_metadata_list), desc="Processing images"):
            image_id_for_progress = future_to_entry[future]
            try:
                result = future.result()
                if result:
                    image_id, conditions = result
                    all_conditions_map[image_id] = conditions
            except Exception as e:
                print(f"Error in processing future for image_id '{image_id_for_progress}': {e}")
    
    # Save results
    if all_conditions_map:
        save_conditions_data(all_conditions_map, args.output_file)
    else:
        print("No conditions were successfully processed or collected.")

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to get conditions using OpenAI API.")
    parser.add_argument(
        "parent_path",
        type=str,
        help="Parent directory containing the metadata file and image files/subdirectories."
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.jsonl",
        help="Name of the metadata JSONL file within the parent_path (default: metadata.jsonl)."
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="conditions/preprocess/refined_conditions.json",
        help="Path to the JSON file containing the list of questions."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="conditions/seen/vlm_conditions_output.json",
        help="Path to save the output JSON file with conditions."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None, # Defaulting to None, will try os.getenv first
        help="OpenAI API key. If not provided, attempts to use OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_OPENAI_MODEL})."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_API_RETRIES,
        help=f"Maximum number of retries for API calls (default: {DEFAULT_MAX_API_RETRIES})."
    )
    parser.add_argument(
        "--api_delay",
        type=float,
        default=DEFAULT_API_REQUEST_DELAY_SECONDS,
        help=f"Base delay in seconds between API retries (default: {DEFAULT_API_REQUEST_DELAY_SECONDS})."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of worker threads for concurrent processing (default: {DEFAULT_MAX_WORKERS})."
    )

    parsed_args = parser.parse_args()
    main(parsed_args)