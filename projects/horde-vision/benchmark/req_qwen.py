import asyncio
import aiohttp
import base64
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
USER_PROMPT = "Бұл суретті сипаттаңыз."
BATCH_SIZE = 16  # Batch size for processing images
MAX_CONCURRENT_REQUESTS = 16  # Limit total concurrent requests
OUTPUT_FILE = "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025_qwenvl25_captions_rq.json"
TIMEOUT_PER_REQUEST = 60  # Timeout for each image processing request (seconds)

# Load dataset
with open(
    "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025.json", "r"
) as f:
    dataset = json.load(f)

# Create base directory for images
base_image_path = "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025/"


# Function to encode image to base64
async def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


# Function to process a single image
async def process_image(client, image_path, prompt, timeout=TIMEOUT_PER_REQUEST):
    try:
        base64_image = await encode_image(image_path)
        if not base64_image:
            return None

        start_time = time.time()
        # Remove the unused ClientSession
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=256,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                            "min_pixels": 64 * 64,
                            "max_pixels": 1280 * 1280,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            timeout=timeout,
        )
        processing_time = time.time() - start_time
        return {
            "response": response.choices[0].message.content,
            "processing_time": processing_time,
        }
    except asyncio.TimeoutError:
        print(f"Timeout processing image {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Process images in batches
async def process_batch(client, image_batch, prompt):
    tasks = [process_image(client, image_path, prompt) for image_path in image_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Main function to process all images
async def process_all_images():
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    # Prepare image paths
    image_paths = []
    image_keys = []
    for image_key in dataset.keys():
        full_image_path = os.path.join(base_image_path, Path(image_key))
        image_paths.append(full_image_path)
        image_keys.append(image_key)
    # image_paths = image_paths[:32]  # Limit to 8 images for testing
    # image_keys = image_keys[:32]

    # Create batches
    batches = [
        image_paths[i : i + BATCH_SIZE] for i in range(0, len(image_paths), BATCH_SIZE)
    ]
    print(f"Num batches: {len(batches)}")
    # Process batches with progress bar
    results = []
    batch_indices = []

    for batch_idx, batch in enumerate(tqdm_asyncio(batches, desc="Processing batches")):
        print(f"Processing batch of {len(batch)} images")
        batch_result = await process_batch(client, batch, USER_PROMPT)

        # Keep track of indices for proper mapping
        for i, result in enumerate(batch_result):
            if not isinstance(result, Exception) and result is not None:
                results.append(result)
                batch_indices.append(batch_idx * BATCH_SIZE + i)

        print(f"Completed batch of {len(batch)} images")

    # Create output data with proper indexing
    output_data = {}
    for idx, original_idx in enumerate(batch_indices):
        if original_idx < len(image_keys):
            output_data[image_keys[original_idx]] = results[idx]["response"]

    # Calculate statistics
    processing_times = [
        result["processing_time"] for result in results if result is not None
    ]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    total_images = len(results)

    print(f"Processed {total_images} images")
    print(f"Average processing time per image: {avg_time:.2f} seconds")

    # Fix potential division by zero
    if processing_times:
        total_processing_time = sum(processing_times)
        throughput = (
            total_images / (total_processing_time / BATCH_SIZE)
            if total_processing_time > 0
            else 0
        )
        print(f"Total throughput: {throughput:.2f} images/second")
    else:
        print("No images were successfully processed")

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_FILE}")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(process_all_images())
