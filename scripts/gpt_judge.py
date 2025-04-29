import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

JUDGING_DATA_PATH = "captions_output.json"
OUTPUT_PATH = "judged_captions_output.json"
MODEL_NAME = "gpt-4o-mini-2024-07-18"

def judge_captions(client: OpenAI, input_path: str, output_path: str, model: str = MODEL_NAME):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = {}

    for image_id, entry in data.items():
        image_path = entry["image_path"]
        caption_1 = entry["caption_1"]
        caption_2 = entry["caption_2"]

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"[ERROR] Failed to read: {image_path}. Reason: {e}")
            continue

        print(f"Judging: {image_path}")

        keyword = Path(image_path).parts[-2]
        judge_prompt = f"""
You are a visual evaluator comparing two Kazakh captions (caption_1, caption_2) for an image tagged with '{keyword}'.
Choose the one that best matches only visible content—no guesses, opinions, or stylistic bias.
Prioritize accuracy, completeness, relevance. Penalize hallucinations, omissions, and subjectivity.
Reply ONLY with: 1 or 2 (just the number, nothing else).
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": judge_prompt},
                        {"type": "text", "text": f"caption_1: {caption_1}"},
                        {"type": "text", "text": f"caption_2: {caption_2}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
                    ]},
                ],
                max_tokens=10,
            )
            content = response.choices[0].message.content.strip()
            print(f"[RESULT] Judgement: {content}")

            if content == "1":
                winner = "with_keyword"
            elif content == "2":
                winner = "without_keyword"
            else:
                winner = "error"

        except Exception as e:
            print(f"[ERROR] OpenAI judging failed on {image_path}: {e}")
            winner = "error"

        updated_data[image_id] = {
            "image_path": image_path,
            "caption_1": caption_1,
            "caption_2": caption_2,
            "winner": winner,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"\n[FINISH] Judging complete. Saved to {output_path}")

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()
    judge_captions(client, JUDGING_DATA_PATH, OUTPUT_PATH)
