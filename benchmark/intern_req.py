import asyncio
import aiohttp
import aiofiles
import base64
import json
import os
import time
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Configuration
OPENAI_API_BASE = "http://localhost:23333/v1"
MODEL = "OpenGVLab/InternVL3-8B"
MODEL_NAME_FOR_SAVE = "internvl3"
MAX_CONCURRENT_REQUESTS = 32
BATCH_SIZE = 32
TIMEOUT_PER_REQUEST = 10
USER_PROMPT = "Бұл суретті қазақ тілінде сипаттаңыз."

FILES = [
    ("/workspace/benchmarks/benchmark_caption.json", "caption"),
    ("/workspace/benchmarks/benchmark_instruct_follow.json", "instruction"),
    ("/workspace/benchmarks/benchmark_ocr.json", "instruction"),
    ("/workspace/benchmarks/benchmark_reason.json", "instruction"),
    ("/workspace/benchmarks/benchmark_vqa.json", "vqa"),
]


# Utils
async def encode_image(image_path):
    try:
        async with aiofiles.open(image_path, "rb") as f:
            content = await f.read()
        return base64.b64encode(content).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


async def get_prediction(session, semaphore, image_path, prompt):
    async with semaphore:
        try:
            new_path = "/workspace/valid_images_0408/" + image_path.split('/')[-1]
            base64_image = await encode_image(new_path)
            if not base64_image:
                return None

            headers = {"Content-Type": "application/json"}
            payload = {
                "model": MODEL,
                "max_tokens": 160,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            }

            async with session.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                try:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"❌ Request failed for {image_path}: {resp.status} - {error_text}")
                        return None

                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]
                finally:
                    await resp.release()  # Ensure the stream is properly closed

        except asyncio.TimeoutError:
            print(f"⏱️ Timeout processing {image_path}")
            return None
        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")
            return None


async def process_items(session, semaphore, items, input_dir, task_type):
    async def process_item(item):
        image_path = os.path.join(input_dir, item["image_path"])
        prompt = USER_PROMPT if task_type == "caption" else item.get("instruction", "")
        if task_type != "caption":
            prompt += " Жауабын қазақ тілінде жазыңыз."
        prediction = await get_prediction(session, semaphore, image_path, prompt)
        item["prediction"] = prediction
        return item

    batched = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    results = []
    for batch in tqdm_asyncio(batched, desc=f"Processing {task_type}"):
        tasks = [process_item(item) for item in batch]
        results.extend(await asyncio.gather(*tasks))
    return results


async def process_vqa_items(session, semaphore, items, input_dir):
    async def process_qa(item, qa):
        image_path = os.path.join(input_dir, item["image_path"])
        prompt = qa["question"] + " Жауабын қазақ тілінде жазыңыз."
        prediction = await get_prediction(session, semaphore, image_path, prompt)
        qa["prediction"] = prediction

    batched_items = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    for batch in tqdm_asyncio(batched_items, desc="Processing VQA"):
        tasks = []
        for item in batch:
            for qa in item["qa_pairs"]:
                tasks.append(process_qa(item, qa))
        await asyncio.gather(*tasks)
    return items


async def process_file(session, file_path, task_type):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    input_dir = os.path.dirname(file_path)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    if task_type == "vqa":
        result = await process_vqa_items(session, semaphore, data, input_dir)
    else:
        result = await process_items(session, semaphore, data, input_dir, task_type)

    output_path = file_path.replace(".json", f"_predicted_{MODEL_NAME_FOR_SAVE}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved predictions to {output_path}")


# Entrypoint
async def main():
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=60)  # Total time per request
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for file_name, task_type in FILES:
            print(f"\n📂 Processing {file_name} [{task_type}]")
            await process_file(session, file_name, task_type)


if __name__ == "__main__":
    asyncio.run(main())