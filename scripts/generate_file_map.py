import json
import unicodedata
from urllib.parse import quote
from pathlib import Path

LOCAL_ROOT = Path("/home/user/kz-mm/data/full_dataset").resolve()
REPO_ID = "horde-research/kaz-vision-50k"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
OUTPUT_JSON = "file_map.json"

def generate_file_map(local_root: Path, base_url: str) -> dict:
    file_map = {}

    for file_path in local_root.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_root)
            normalized_path = unicodedata.normalize("NFD", relative_path.as_posix())
            encoded_path = quote(normalized_path)
            hf_url = f"{base_url}/{encoded_path}"
            file_map[str(file_path)] = hf_url

    return file_map

def save_file_map(file_map: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(file_map, f, ensure_ascii=False, indent=2)
    print(f"Mapping completed: {len(file_map)} files saved to {output_path}")

if __name__ == "__main__":
    file_map = generate_file_map(LOCAL_ROOT, BASE_URL)
    save_file_map(file_map, OUTPUT_JSON)
