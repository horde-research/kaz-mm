import json
import base64
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI


def get_random_path_excluding_first(paths: list[str]) -> str:
    """
    Auxilary function for choosing random image from dataset
    """
    if len(paths) <= 1:
        return paths[0]
    return random.choice(paths[1:])


def generate_kz_captions(
    client: OpenAI,
    input_json_path: str,
    output_json_path: str,
    model_name: str = "gpt-4o-mini-2024-07-18"
):
    """
    Generate captions for images listed in the input JSON file using GPT, not using batch-mode.
    Saves a dictionary with image path and two captions (with and without keyword) to JSON.
    Args:
        client (OpenAI): Initialized OpenAI client.
        input_json_path (str): Path to the input JSON file with image paths.
        output_json_path (str): Path to save the output JSON with captions.
        model_name (str): Name of the model to use.
    """
    with open(input_json_path, "r") as f:
        image_paths = json.load(f)

    # To group by categories 
    grouped = defaultdict(list)
    for path_str in image_paths:
        path = Path(path_str)
        parts = path.parts
        try:
            key = f"{parts[-4]}/{parts[-3]}"
            grouped[key].append(path_str)
        except IndexError:
            print(f"Wrong path: {path_str}")
            continue

    result = {}
    
    for idx, paths in enumerate(grouped.values(), 1):      
        image_path = paths[0]
        if idx > 50:
            break
        image_path = get_random_path_excluding_first(paths)
        topic_name = Path(image_path).parts[-2]

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        prompt_with_keyword = f"""
You are a precise visual assistant for a blind person. Write a factual, detailed caption in Kazakh for the image, using exactly five sentences (40–60 words). Describe only what is directly visible: objects, their appearance, interactions, and environment. Use clear, neutral Kazakh; avoid style, opinions, or guesses (e.g., no “looks cozy” or “seems happy”). No 'Суретте' at the beginning. Just describe the image directly. 
Include '{topic_name}' only if they clearly match the scene—don’t force them.
Each sentence must describe a distinct, observable detail, such as:
 • Main objects (color, size, shape, position)
 • Related objects nearby or interacting
 • People/animals (visible action or pose only)
 • Background (e.g., furniture, walls, floor)
 • Lighting or visible conditions
Use simple, natural Kazakh and precise, factual adjectives (e.g., “үлкен,” “көк”)—not subjective (e.g., “әдемі,” “жылы”). Don’t mention time, purpose, or unseen context.
        """

        prompt_without_keyword = """
You are a precise visual assistant. Write a factual, detailed caption in Kazakh for the image, using exactly five sentences (40–60 words).
Describe only what is directly visible: objects, their appearance, interactions, and environment. Use clear, neutral Kazakh; avoid style, opinions, or guesses (e.g., no “looks cozy” or “seems happy”).
Each sentence must describe a distinct, observable detail, such as:
 • Main objects (color, size, shape, position)
 • Related objects nearby or interacting
 • People/animals (visible action or pose only)
 • Background (e.g., furniture, walls, floor)
 • Lighting or visible conditions
Use simple, natural Kazakh and precise, factual adjectives (e.g., “үлкен,” “көк”)—not subjective (e.g., “әдемі,” “жылы”). Don’t mention time, purpose, or unseen context.
        """

        # With keyword
        try:
            response_1 = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_with_keyword},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            caption_with_kw = response_1.choices[0].message.content.strip()
            print(f"\n[DEBUG] Response with keyword:\n{caption_with_kw!r}\n")
        except Exception as e:
            print(f"Error (with keyword) on {image_path}: {e}")
            caption_with_kw = "[ERROR]"

        # Without keyword
        try:
            response_2 = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_without_keyword},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            caption_without_kw = response_2.choices[0].message.content.strip()
            print(f"\n[DEBUG] Response without keyword:\n{caption_without_kw!r}\n")
        except Exception as e:
            print(f"Error (without keyword) on {image_path}: {e}")
            caption_without_kw = "[ERROR]"
        
        result[f"image_with_caption_{idx}"] = {
            "image_path": image_path,
            "caption_1": caption_with_kw,
            "caption_2": caption_without_kw,
        }
        

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(result)} entries to {output_json_path}")


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()

    input_json_path = "train_set_Xsmall.json"
    output_json_path = "captions_output.json"

    generate_kz_captions(
        client=client,
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        model_name="gpt-4o-mini-2024-07-18"
    )
