import json
from pathlib import Path

INPUT_JSON_PATH = "valid_set_Xsmall.json"
LINKS_JSON_PATH = "file_map.json"
OUTPUT_BATCH_JSONL = "batch_input.jsonl"
MAX_ITEMS = 50_000

def generate_batch_requests(input_paths_file, links_dict_file, max_items=MAX_ITEMS):
    with open(input_paths_file, "r") as f:
        input_paths = json.load(f)

    with open(links_dict_file, "r") as f:
        url_lookup = json.load(f)

    batch_requests = []

    for idx, path_str in enumerate(input_paths, 1):
        if idx > max_items:
            break

        path = Path(path_str)
        try:
            topic_name = path.parts[-2]
        except IndexError:
            print(f"[WARN] Skipped due to path error: {path_str}")
            continue

        image_url = url_lookup.get(path_str)
        if not image_url:
            print(f"[WARN] No URL found for: {path_str}")
            continue

        prompt = f"""
You are a precise visual assistant for a blind person. Write a factual, detailed caption in Kazakh for the image, using exactly five sentences (40–60 words). Describe only what is directly visible: objects, their appearance, interactions, and environment. Use clear, neutral Kazakh; avoid style, opinions, or guesses (e.g., no “looks cozy” or “seems happy”). No 'Суретте' at the beginning. Just describe the image directly. 
Include '{topic_name}' only if they clearly match the scene—don’t force them.
Each sentence must describe a distinct, observable detail, such as:
 • Main objects (color, size, shape, position)
 • Related objects nearby or interacting
 • People/animals (visible action or pose only)
 • Background (e.g., furniture, walls, floor)
 • Lighting or visible conditions
Use simple, natural Kazakh and precise, factual adjectives (e.g., “үлкен,” “көк”)—not subjective (e.g., “әдемі,” “жылы”). 
Don’t mention time, purpose, or unseen context.
        """

        batch_requests.append({
            "custom_id": f"caption_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini-2024-07-18",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
        })

    return batch_requests

def save_to_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} entries to {path}")

if __name__ == "__main__":
    requests = generate_batch_requests(INPUT_JSON_PATH, LINKS_JSON_PATH)
    save_to_jsonl(requests, OUTPUT_BATCH_JSONL)
