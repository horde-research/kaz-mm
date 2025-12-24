import json
import glob
import re

# Task types and output files mapping
TASKS = {
    "caption": "caption_task_eval.json",
    "vqa": "vqa_task_eval.json",
    "ocr": "ocr_task_eval.json",
    "reason": "reason_task_eval.json",
    "instruct_follow": "inst_follow_task_eval.json",
}

def extract_custom_id_info(custom_id):
    """Extract task_type, number, and model from custom_id.
    Example: 'gemma3_caption_1' -> ('caption', 1, 'gemma3')
    """
    for task in TASKS:
        pattern = rf"^(.+?)_{task}_(\d+)$"
        match = re.match(pattern, custom_id)
        if match:
            return task, int(match.group(2)), match.group(1)
    return None, None, None

def extract_score(task_type, content_str):
    """Parse JSON content and extract score."""
    try:
        data = json.loads(content_str)
        if task_type == "caption":
            return float(data["overall_score"])
        return float(data["score"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Failed to parse score: {e}")

def process_files():
    """Process all files and align results by position (1st result ↔ 1st result, etc.)"""
    # Temporary storage: model → task → list of (original_num, score)
    per_model = {}

    errors = []

    for file_path in glob.glob("output_eval_results_*.json"):
        print(f"Processing: {file_path}")

        try:
            with open(file_path, "r") as f:
                payloads = json.load(f)
        except Exception as e:
            print(f" ✗ Corrupted file: {e}")
            continue

        current_model = None

        for payload in payloads:
            custom_id = payload.get("custom_id")
            if not custom_id:
                errors.append({"error": "missing custom_id", "payload": payload})
                continue

            task_type, raw_num, model = extract_custom_id_info(custom_id)
            if not task_type:
                errors.append({"error": "unparsed custom_id", "custom_id": custom_id})
                continue

            # Detect model name from first payload in file (or extract from filename if needed)
            if current_model is None:
                current_model = model
            elif model != current_model:
                print(f"Warning: mixed models in {file_path}")
                
            try:
                content = payload["response"]["body"]["choices"][0]["message"]["content"]
                score = extract_score(task_type, content)
            except Exception as e:
                errors.append({
                    "error": str(e),
                    "custom_id": custom_id,
                    "model": model
                })
                continue

            if model not in per_model:
                per_model[model] = {task: [] for task in TASKS}

            per_model[model][task_type].append((raw_num, score))

    # Now align by sorted order per task
    results = {task: {} for task in TASKS}

    for task in TASKS:
        # Get all models that have this task
        models_with_task = [m for m in per_model if task in per_model[m] and per_model[m][task]]

        if not models_with_task:
            continue

        # Sort each model's results by their original raw_num (to restore original dataset order)
        for model in models_with_task:
            per_model[model][task].sort(key=lambda x: x[0])  # sort by raw_num
            per_model[model][task] = [score for _, score in per_model[model][task]]

        # Determine how many samples we have (take the minimum to be safe)
        lengths = [len(per_model[m][task]) for m in models_with_task]
        n_samples = min(lengths)

        print(f"{task}: aligning {n_samples} samples across {len(models_with_task)} models")

        for i in range(n_samples):
            key = f"{task}_{i+1}"  # 1-based indexing
            results[task][key] = {}
            for model in models_with_task:
                results[task][key][model] = per_model[model][task][i]

    return results, errors

def save_results(results, errors):
    """Save aggregated results to JSON files."""
    for task, output_file in TASKS.items():
        data = results[task]
        # Sort by task number
        sorted_data = dict(sorted(
            data.items(),
            key=lambda x: int(x[0].split("_")[-1])
        ))
        
        with open(output_file, "w") as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)
        print(f"✓ {output_file}")

    with open("error_samples.json", "w") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    if errors:
        print(f"✓ error_samples.json ({len(errors)} errors)")

if __name__ == "__main__":
    print("Starting aggregation...\n")
    results, errors = process_files()
    print("\nSaving results...\n")
    save_results(results, errors)
    print("\n✔ Aggregation completed")