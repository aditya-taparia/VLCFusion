from PIL import Image
import numpy as np
import torch
import re
import json

from openai import OpenAI
import base64
from io import BytesIO

import os

parent_path = "YOUR_PATH_HERE"  # Replace with your actual path
vis_file_path = f"{parent_path}/metadata.jsonl"

with open(vis_file_path, "r") as file:
    lines = file.readlines()
vis_data = []
for line in lines:
    vis_data.append(json.dumps(json.loads(line)))
    
questions_file = 'conditions/preprocess/refined_conditions.json'

with open(questions_file, 'r') as f:
    questions_list = json.load(f)
    
# formatted_questions = "\n".join(["- " + question for question in questions_list])
formatted_questions = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions_list)])

client = OpenAI(
    api_key="YOUR_API_KEY"
)

def get_conditions(image, retry_count=0, max_retries=3):   
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        if image.dtype in [np.float32, np.float64]:
            image = (image * 255).astype(np.uint8)

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        try:
            image = Image.fromarray(image)
        except Exception as e:
            raise ValueError(
                "Failed to convert numpy array to PIL image. Ensure the array has the correct shape and dtype."
            ) from e
    
    def encode_image(image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0) 
        return base64.b64encode(buffer.read()).decode("utf-8")

    image_encoded = encode_image(image)
    conditions = []
    
    message_text = (
        "Answer the following questions based on the given image:\n"
        "## Questions:\n"
        f"{formatted_questions}\n\n"
        f"IMPORTANT: Your answer must be a JSON object with exactly {len(questions_list)} keys. "
        f"The keys should be the numbers from 1 to {len(questions_list)} (as strings) and each value must be a boolean (True or False), one for each question, and nothing else. "
        "The image is provided after these questions."
    )
    
    response = client.chat.completions.create(
        # model="gpt-4o",
        model="gpt-4o-2024-11-20",
        messages=[
            {
            "role": "system",
            "content": (
                    "You are a highly specialized assistant that provides concise answers to specific questions about images. "
                    "For each question, respond with either True or False only. "
                    "Provide your answer as a JSON object with keys corresponding to the question numbers (from 1 to "
                    f"{len(questions_list)}). Do not provide additional context or descriptions."
                )
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": message_text,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_encoded}",
                },
                },
            ],
            }
        ],
        )
    
    if response.choices[0].message.content is None:
        return get_conditions(image, retry_count=retry_count, max_retries=max_retries)
    
    response_text = response.choices[0].message.content.strip()
    
    response_text = re.sub(r"```(json)?", "", response_text).strip()
    response_text = re.sub(r"```", "", response_text).strip()
    try:
        conditions = json.loads(response_text)
    except json.JSONDecodeError:
        conditions_list = [item.strip() for item in response_text.strip("[]").split(",")]
        conditions = {str(i+1): (True if str(item).strip().lower() == "true" else False)
                      for i, item in enumerate(conditions_list)}
    
    if not isinstance(conditions, dict) or len(conditions.keys()) != len(questions_list):
        if retry_count < max_retries:
            print(f"Mismatch in response structure (got {len(conditions) if isinstance(conditions, dict) else 'non-dict'} vs expected {len(questions_list)}). Retrying {retry_count+1}/{max_retries}...")
            time.sleep(0.5)
            return get_conditions(image, retry_count=retry_count+1, max_retries=max_retries)
        else:
            # raise ValueError("Response structure does not match the number of questions even after retries.")
            print("Response structure does not match the number of questions even after retries.")
            return []
    
    sorted_conditions = [conditions[str(i+1)] for i in range(len(questions_list))]
    return sorted_conditions

conditions = {}

import time

def process_image(data):
    image_id = data["image_id"]
    image_path = data["file_name"]
    image = Image.open(os.path.join(parent_path, image_path)).convert('RGB')
    condition = get_conditions(image)
    time.sleep(0.75)
    
    return image_id, condition

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for data in tqdm(vis_data, desc="Processing images"):
        data = json.loads(data)
        future = executor.submit(process_image, data)
        futures.append(future)

    for future in tqdm(futures, desc="Collecting results"):
        image_id, condition = future.result()
        conditions[image_id] = condition
        
# Save the conditions to a JSON file
with open("conditions/seen/vlm_test.json", "w") as file:
    json.dump(conditions, file, indent=4)