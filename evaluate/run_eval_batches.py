import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# -------- CONFIG -------- #
BATCH_INPUT_FOLDER = "."        # folder with files
BATCH_INPUT_PATTERN = "captions_ocr_batch_eval_llama.jsonl"  # READ ALL your generated prompt files
OUTPUT_FOLDER = "."
COMPLETION_WINDOW = "24h"

load_dotenv()
client = OpenAI()


def submit_batch(input_path: Path):
    """Upload file and create batch job. Returns batch_id, task_name."""
    # Detect task name from filename: captions_caption_batch_eval → caption
    parts = input_path.stem.split("_")
    # Example: 'captions_caption_batch_eval' → ['captions', 'caption', 'batch', 'eval']
    task_name = parts[1] if len(parts) > 1 else input_path.stem

    batch_name = f"evaluate_kz_data_{task_name}"

    print(f"\n--- SUBMITTING BATCH: {task_name.upper()} ---")

    try:
        with open(input_path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")

        print(f"[UPLOAD] {input_path.name} → File ID {upload.id}")

    except Exception as e:
        print(f"[ERROR] Upload failed for {input_path.name}: {e}")
        return None, None

    try:
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=COMPLETION_WINDOW,
            metadata={"name": batch_name}
        )

        print(f"[BATCH] Created {batch.id} | Status: {batch.status}")
        return batch.id, task_name

    except Exception as e:
        print(f"[ERROR] Failed to create batch: {e}")
        return None, None


def download_results(batch, task_name, output_folder):
    """Download results for a completed batch."""
    file_id = batch.output_file_id
    output_path = output_folder / f"output_eval_{task_name}_fix.json"

    if not file_id:
        print(f"[ERROR] Batch {batch.id} has no output_file_id.")
        return

    print(f"[DOWNLOAD] {batch.id} → File {file_id}")

    response = client.files.content(file_id)

    try:
        lines = [json.loads(line) for line in response.text.strip().splitlines()]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Saved {len(lines)} items → {output_path.name}")

    except Exception as e:
        print(f"[ERROR] Failed parsing results for {batch.id}: {e}")


def monitor_batches(batch_map, output_folder_path):
    """Monitor all batch jobs until they finish."""
    remaining = set(batch_map.keys())

    print("\n--- MONITORING ALL BATCH JOBS ---")

    while remaining:
        time.sleep(15)

        for batch_id in list(remaining):
            batch = client.batches.retrieve(batch_id)
            task_name = batch_map[batch_id]["task"]

            print(f"[STATUS] {batch_id} ({task_name}): {batch.status}")

            if batch.status in {"completed", "failed", "expired"}:
                remaining.remove(batch_id)

                if batch.status == "completed":
                    download_results(batch, task_name, output_folder_path)
                else:
                    print(f"[INFO] Batch {batch_id} finished with status: {batch.status}")


# -------- MAIN -------- #

if __name__ == "__main__":

    input_folder_path = Path(BATCH_INPUT_FOLDER)
    output_folder_path = Path(OUTPUT_FOLDER)

    batch_files = sorted(input_folder_path.glob(BATCH_INPUT_PATTERN))

    if not batch_files:
        print("[ERROR] No matching batch files found.")
        exit()

    print("\nFound batch prompt files:")
    for f in batch_files:
        print(" -", f.name)

    batch_map = {}

    # Submit all batches at once
    for file_path in batch_files:
        batch_id, task = submit_batch(file_path)
        if batch_id:
            batch_map[batch_id] = {"task": task}

    monitor_batches(batch_map, output_folder_path)