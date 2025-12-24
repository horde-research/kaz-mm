import time
import json
from openai import OpenAI
from dotenv import load_dotenv

BATCH_INPUT_PATH = "batch_input.jsonl"
BATCH_NAME = "generate_kz_captions_batch"
COMPLETION_WINDOW = "24h"
OUTPUT_JSON = "output_valid_dataset_xsmall.json"

def run_batch(input_path: str, output_path: str):
    load_dotenv()
    client = OpenAI()

    with open(input_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"[UPLOAD] File uploaded: {upload.id}")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window=COMPLETION_WINDOW,
        metadata={"name": BATCH_NAME}
    )
    
    print(f"[BATCH] Batch created: {batch.id} | Status: {batch.status}")

    while True:
        status = client.batches.retrieve(batch.id)
        print(f"[WAIT] Status: {status.status}")
        if status.status in {"completed", "failed", "expired"}:
            break
        time.sleep(15)

    if status.status == "completed":
        file_id = status.output_file_id
        if not file_id:
            print("[ERROR] Batch completed, but output_file_id is missing.")
            return

        print(f"[DOWNLOAD] Fetching content from file ID: {file_id}")
        response = client.files.content(file_id)

        lines = [json.loads(line) for line in response.text.strip().splitlines()]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Saved {len(lines)} entries to {output_path}")

    elif status.status == "failed":
        print(f"[FAILURE] Batch failed. Error: {status.error}")
    else:
        print(f"[INFO] Final status: {status.status}")

if __name__ == "__main__":
    run_batch(BATCH_INPUT_PATH, OUTPUT_JSON)
