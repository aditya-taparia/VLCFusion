# %%
from openai import OpenAI
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import torch
import re
import json

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import mmengine

# %%
client = OpenAI(
    api_key="YOUR_API_KEY",
)

# %%
questions_file = 'YOUR_PATH/refined_conditions.json'

with open(questions_file, 'r') as f:
    questions_list = json.load(f)
    
# formatted_questions = "\n".join(["- " + question for question in questions_list])
formatted_questions = "\n".join([f"{i+1}. {question}" for i, question in enumerate(questions_list)])

# %%
root_path = 'YOUR_PATH/LidarTraining/night-day-frames/training/image_0/'
anno_est = 'YOUR_PATH/LidarTraining/night-day-frames/waymo_infos_val.pkl'
data_list = mmengine.load(anno_est)['data_list']

# %%
def get_conditions(image, retry_count=0, max_retries=3):
    """ 
    Get the vlm conditions from the image
    """
    
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
    
    try:
        response = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4o-2024-11-20",
            messages=[
                {
                "role": "system",
                # "content": "You are a highly specialized assistant that provides concise answers to specific questions about images. For each question, respond with either True or False only. Provide your answer as a JSON list of booleans. Do not provide additional context or descriptions."
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
                    # "Answer the following questions based on the given image:\n## Questions:\n"
                    #         "- Do the input image depicts daytime based on the lighting conditions?\n"
                    #         "- Is the weather clear in the input (not raining)?\n"
                    #         "- Is the input image taken in a urban area?\n"
                    #         "- Image:\n"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_encoded}",
                    },
                    },
                ],
                }
            ]
        )
    except Exception as e:
        if retry_count < max_retries:
            print(f"Failed to get response. Retrying {retry_count+1}/{max_retries}...")
            time.sleep(0.5)
            return get_conditions(image, retry_count=retry_count+1, max_retries=max_retries)
        else:
            print("Failed to get response even after retries.")
            return []
    
    if response.choices[0].message.content is None:
        return get_conditions(image, retry_count=retry_count, max_retries=max_retries)
    
    response_text = response.choices[0].message.content.strip()
    
    response_text = re.sub(r"```(json)?", "", response_text).strip()
    response_text = re.sub(r"```", "", response_text).strip()
    try:
        conditions = json.loads(response_text)
    except json.JSONDecodeError:
        # conditions = [item.strip() for item in response_text.strip("[]").split(",")]
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

# %%
import time

def process_img_info(data):
    img_idx = data['sample_idx']
    
    img_path = data['images']['CAM_FRONT']['img_path']
    img_path = root_path + img_path
    img = Image.open(img_path).convert('RGB')

    try:
        conditions = get_conditions(img)
    except Exception as e:
        print(f"Failed to get conditions for image {img_idx}. Error: {e}")
        conditions = []
    
    # Sleep to avoid rate limit
    time.sleep(0.75)

    return {
        'image_idx': img_idx,
        'image_path': img_path,
        'conditions': conditions,
    }

# %%
jsonl_data = []

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(tqdm(executor.map(process_img_info, data_list), total=len(data_list)))

jsonl_data.extend(results)

# %%
# Check the length of conditions for each data point
for data in jsonl_data:
    if len(data['conditions']) != len(questions_list):
        print(f"Image {data['image_idx']} has {len(data['conditions'])} conditions instead of {len(questions_list)}")

# %%
# Save the data to a jsonl file
with open('YOUR_PATH/night-day_validation.jsonl', 'w') as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + '\n')