import json
import random
from pathlib import Path
from collections import Counter

INPUT_JSONL = "/workspace/train_data_folder/final_train_results2411_with_local_paths_SFT.jsonl"
OUTPUT_JSON = "/workspace/train_data_folder/train_SFT.json"   # ← changed to .json

# ---- Adjustable ratios ----
CAPTION_RATIO = 0.5
OTHER_TASKS = ["vqa", "ocr", "reason", "instruct_follow"]
OTHER_RATIO = (1 - CAPTION_RATIO) / len(OTHER_TASKS)

TASK_PROBS = {
    "caption": CAPTION_RATIO,
    "vqa": OTHER_RATIO,
    "ocr": OTHER_RATIO,
    "reason": OTHER_RATIO,
    "instruct_follow": OTHER_RATIO,
}

def build_conversation(instruction, answer, image_path):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_path},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
    }

def sample_task():
    tasks = list(TASK_PROBS.keys())
    probs = list(TASK_PROBS.values())
    return random.choices(tasks, weights=probs, k=1)[0]

def convert_entry(entry):
    img = entry["local_image_path"]
    task = sample_task()

    if task == "caption":
        instruction = "Бұл суретті қазақ тілінде сипаттаңыз."
        answer = entry["caption"]["text"]
        return build_conversation(instruction, answer, img), "caption"

    elif task == "vqa":
        qa_item = random.choice(entry["vqa"])
        instruction = qa_item["question"]
        answer = qa_item["answer"]
        return build_conversation(instruction, answer, img), "vqa"

    elif task == "ocr":
        instruction = entry["ocr"]["instruction"]
        answer = entry["ocr"]["answer"]
        return build_conversation(instruction, answer, img), "ocr"

    elif task == "reason":
        instruction = entry["reason"]["instruction"]
        answer = entry["reason"]["answer"]
        return build_conversation(instruction, answer, img), "reason"

    elif task == "instruct_follow":
        instruction = entry["instruct_follow"]["instruction"]
        answer = entry["instruct_follow"]["answer"]
        return build_conversation(instruction, answer, img), "instruct_follow"

    raise ValueError("Unknown task")


def main():
    task_stats = Counter()
    converted_data = []   # ← We collect everything in a list

    print("Reading and converting...")
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"   processed {i} lines...")

            entry = json.loads(line.strip())
            conv, task = convert_entry(entry)
            converted_data.append(conv)
            task_stats[task] += 1

    # Write ONE valid JSON file
    print(f"Writing {len(converted_data):,} examples to {OUTPUT_JSON} ...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)  # indent optional, makes it human-readable

    print("\n=== Task distribution ===")
    total = sum(task_stats.values())
    for k, v in task_stats.most_common():
        print(f"{k:15} : {v:,}  ({v/total*100:5.2f}%)")
    print("=========================\n")
    print(f"Successfully saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()