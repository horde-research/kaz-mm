import asyncio
import aiofiles
import base64
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from PIL import Image
import io
# Config
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

#vllm serve /workspace/train_data_folder/lora128_vision_layer_cosine_2e-4_bs72_ep1_0412_145218 --port 8000   --host 0.0.0.0
#vllm serve /workspace/train_data_folder/lora128_vision_layer_cosine_2e-4_bs72_ep1_0412_145218 --port 8000   --host 0.0.0.0
#vllm serve /workspace/train_data_folder/lora64_vision_layer_linear_2e-4_bs96_ep2_0512_155341 --port 8000   --host 0.0.0.0

MODEL = "my-kazakh-lora"
MODEL_NAME_FOR_SAVE = "RL2-qwen-lora64-1ep"

#MODEL = "/workspace/train_data_folder/lora64_vision_layer_linear_2e-4_bs96_ep2_0512_155341"
#MODEL_NAME_FOR_SAVE = "qwen-lora64-bs96-ep2"

#MODEL = "/workspace/train_data_folder/lora64_vision_layer_cosine_2e-4_bs96_ep1_0412_044414"
#MODEL_NAME_FOR_SAVE = "qwen-lora64-bs96"

# MODEL = "/workspace/train_data_folder/lora128_vision_layer_cosine_2e-4_bs72_ep1_0412_145218"
# MODEL_NAME_FOR_SAVE = "qwen-lora128-bs72"

#MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
#MODEL_NAME_FOR_SAVE = "qwen25"
MAX_CONCURRENT_REQUESTS = 32
BATCH_SIZE = 32
TIMEOUT_PER_REQUEST = 60
USER_PROMPT = "Бұл суретті қазақ тілінде сипаттаңыз."

FILES = [
    ("/workspace/benchmarks/benchmark_caption.json", "caption"),
    ("/workspace/benchmarks/benchmark_instruct_follow.json", "instruction"),
    ("/workspace/benchmarks/benchmark_ocr.json", "instruction"),
    ("/workspace/benchmarks/benchmark_reason.json", "instruction"),
    ("/workspace/benchmarks/benchmark_vqa.json", "vqa"),
]

# Utils
async def encode_image(image_path, max_size=1280):
    """Resize image if it exceeds max_size while maintaining aspect ratio."""
    try:
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        
        # Open image with PIL
        img = Image.open(io.BytesIO(content))
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get current dimensions
        width, height = img.size
        
        # Resize if image is too large
        if max(width, height) > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        
        return base64.b64encode(img_bytes).decode("utf-8")
        
    except Exception as e:
        print(f"Error preparing image {image_path}: {e}")
        return None
# async def encode_image(image_path):
#     try:
#         async with aiofiles.open(image_path, "rb") as f:
#             content = await f.read()
#         return base64.b64encode(content).decode("utf-8")
#     except Exception as e:
#         print(f"Error encoding image {image_path}: {e}")
#         return None

async def get_prediction(client, semaphore, image_path, prompt):
    async with semaphore:
        try:
            new_path = "/workspace/valid_images_0408/" + image_path.split('/')[-1]
            base64_image = await encode_image(new_path)
            if not base64_image:
                return None

            response = await client.chat.completions.create(
                model=MODEL,
                max_tokens=256,
                timeout=TIMEOUT_PER_REQUEST,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            #"min_pixels": 64 * 64,
                            #"max_pixels": 1280 * 1280,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Failed {image_path} with prompt '{prompt}': {e}")
            return None

# Task-specific processors with batching and concurrency
async def process_items(client, semaphore, items, input_dir, task_type):
    async def process_item(item):
        image_path = os.path.join(input_dir, item["image_path"])
        prompt = (
            USER_PROMPT if task_type == "caption" else item.get("instruction", "")
        )
        if task_type != 'caption':
            prompt += " Жауабын қазақ тілінде жазыңыз."
        prediction = await get_prediction(client, semaphore, image_path, prompt)
        item["prediction"] = prediction
        return item

    batched = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    results = []
    for batch in tqdm_asyncio(batched, desc=f"Processing {task_type}"):
        tasks = [process_item(item) for item in batch]
        results.extend(await asyncio.gather(*tasks))
    return results

async def process_vqa_items(client, semaphore, items, input_dir):
    async def process_qa(item, qa):
        image_path = os.path.join(input_dir, item["image_path"])
        prediction = await get_prediction(client, semaphore, image_path, qa["question"] + " Жауабын қазақ тілінде жазыңыз.")
        qa["prediction"] = prediction

    batched_items = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    for batch in tqdm_asyncio(batched_items, desc="Processing VQA"):
        tasks = []
        for item in batch:
            for qa in item["qa_pairs"]:
                tasks.append(process_qa(item, qa))
        await asyncio.gather(*tasks)
    return items

# Dispatcher
async def process_file(client, file_path, task_type):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    #data = data[:16]
    input_dir = os.path.dirname(file_path)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    if task_type == "vqa":
        result = await process_vqa_items(client, semaphore, data, input_dir)
    else:
        result = await process_items(client, semaphore, data, input_dir, task_type)

    output_path = file_path.replace(".json", f"_predicted_{MODEL_NAME_FOR_SAVE}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved predictions to {output_path}")

# Entrypoint
async def main():
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    for file_name, task_type in FILES:
        print(f"\n📂 Processing {file_name} [{task_type}]")
        await process_file(client, file_name, task_type)

if __name__ == "__main__":
    asyncio.run(main())