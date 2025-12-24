import asyncio
import aiohttp
import base64
import json
import os
import time
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Configuration
OPENAI_API_BASE = "http://localhost:23333/v1"
MODEL = "OpenGVLab/InternVL3-8B"
USER_PROMPT = "Бұл суретті сипаттаңыз."
BATCH_SIZE = 16
MAX_CONCURRENT_REQUESTS = 16
TIMEOUT_PER_REQUEST = 60  # seconds
OUTPUT_FILE = "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025_internvl3_captions_rq.json"

# Load dataset
with open(
    "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025.json", "r"
) as f:
    dataset = json.load(f)

# Image folder
base_image_path = "/workspace/kaz-mm/benchmark/valid_dataset_kazvision_26042025/"


# Encode image to base64
async def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


# Process one image
async def process_image(session, image_path, prompt, timeout=TIMEOUT_PER_REQUEST):
    try:
        base64_image = await encode_image(image_path)
        if not base64_image:
            return None

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 256,
        }

        start_time = time.time()
        async with session.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(
                    f"Request failed for {image_path} with status {resp.status}: {text}"
                )
                return None

            result = await resp.json()
            processing_time = time.time() - start_time
            return {
                "response": result["choices"][0]["message"]["content"],
                "processing_time": processing_time,
            }

    except asyncio.TimeoutError:
        print(f"Timeout processing image {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Process a batch of images
async def process_batch(session, image_batch, prompt):
    tasks = [process_image(session, image_path, prompt) for image_path in image_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Process all images
async def process_all_images():
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        image_paths = []
        image_keys = []
        for image_key in dataset.keys():
            full_image_path = os.path.join(base_image_path, Path(image_key))
            image_paths.append(full_image_path)
            image_keys.append(image_key)

        # For testing only
        # image_paths = image_paths[:4]
        # image_keys = image_keys[:4]

        batches = [
            image_paths[i : i + BATCH_SIZE]
            for i in range(0, len(image_paths), BATCH_SIZE)
        ]
        print(f"Num batches: {len(batches)}")

        results = []
        batch_indices = []

        for batch_idx, batch in enumerate(
            tqdm_asyncio(batches, desc="Processing batches")
        ):
            print(f"Processing batch of {len(batch)} images")
            batch_result = await process_batch(session, batch, USER_PROMPT)

            for i, result in enumerate(batch_result):
                if not isinstance(result, Exception) and result is not None:
                    results.append(result)
                    batch_indices.append(batch_idx * BATCH_SIZE + i)

        # Build output mapping
        output_data = {}
        for idx, original_idx in enumerate(batch_indices):
            if original_idx < len(image_keys):
                output_data[image_keys[original_idx]] = results[idx]["response"]

        # Stats
        processing_times = [r["processing_time"] for r in results if r]
        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )
        total_images = len(results)
        print(f"Processed {total_images} images")
        print(f"Average time per image: {avg_time:.2f}s")

        if processing_times:
            total_processing_time = sum(processing_times)
            throughput = total_images / (total_processing_time / BATCH_SIZE)
            print(f"Total throughput: {throughput:.2f} images/sec")
        else:
            print("No images processed successfully")

        # Save output
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Saved results to {OUTPUT_FILE}")


# Run
if __name__ == "__main__":
    asyncio.run(process_all_images())
