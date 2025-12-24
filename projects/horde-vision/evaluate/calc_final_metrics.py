import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ================= CONFIG =================
FINETUNE_DIR = 'results'
OUTPUT_DIR = "results_FINAL"

TASK_FILES = {
    "caption": "caption_task_eval.json",
    "vqa": "vqa_task_eval.json",
    "ocr": "ocr_task_eval.json",
    "ocr": "ocr_task_eval.json",
    "reason": "reason_task_eval.json",
    "instruct_follow": "inst_follow_task_eval.json",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map from internal key to display name
MODEL_MAP = {
    "gemma3": "gemma-3-4b-it", # Corrected for display clarity
    "internvl3": "InternVL3-8B",
    "qwen25": "Qwen2.5-VL-7B-Instruct",
    "qwen3": "Qwen3-VL-8B-Instruct",
    "qolda": "Qolda",
    "RL2-qwen-lora64-1ep": "horde-vision", 
    "llama3-11b-vision": "Llama-3.2-11B-Vision"
}

# The models defined here are the ones that will be included in the final results.
MODEL_COLORS = {
    "gemma3": "#0072B2",      # strong blue
    "internvl3": "#D55E00",   # vermillion / orange
    "qwen25": "#6c3db3",      # dark neutral gray
    "qwen3": "#009E73",       # bluish green
    "qolda": "#F0E442",       # yellow (high luminance)
    "RL2-qwen-lora64-1ep": "#f244a4",  # reddish purple
    "llama3-11b-vision": "#ff7f0e"
}

# Define the set of allowed models based on the MODEL_COLORS keys
ALLOWED_MODELS = set(MODEL_COLORS.keys()) 

# ==========================================

def load_task_data():
    task_data = {}
    all_models = set()
    
    for task_name, filename in TASK_FILES.items():
        filepath = os.path.join(FINETUNE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping {task_name}")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        task_data[task_name] = data
        for sample in data.values():
            all_models.update(sample.keys())
    
    all_models = sorted(list(all_models))
    print(f"Loaded tasks: {list(task_data.keys())}")
    print(f"Models found in files: {all_models}")
    return task_data, all_models

def normalize_if_needed(scores_dict, task_name, allowed_models):
    # Find max possible/observed score in this task
    all_scores = [s for sample in scores_dict.values() 
                  for model, s in sample.items() if model in allowed_models]
    
    if not all_scores:
        return scores_dict, 1.0
    
    max_score = max(all_scores)
    print(f"{task_name}: max observed score = {max_score}")
    
    if max_score < 6.0:
        scale = 100.0 / max_score
        print(f"  → Normalizing to 0–100 scale (×{scale:.2f})")
        for sample in scores_dict.values():
            for model in sample:
                # Only scale the scores of allowed models
                if model in allowed_models:
                    sample[model] = round(sample[model] * scale, 2)
        return scores_dict, scale
    else:
        return scores_dict, 1.0

def compute_metrics(task_data, allowed_models):
    metrics = {}
    normalized_data = {}
    
    model_list = sorted(list(allowed_models)) 

    for task, samples in task_data.items():
        print(f"\nProcessing {task}...")
        
        samples, scale = normalize_if_needed(samples, task, allowed_models)
        normalized_data[task] = samples

        avg_scores = {m: [] for m in model_list} 
        n = len(samples)

        # Collect per-sample scores for allowed models only
        for sample_dict in samples.values():
            for model in model_list: 
                score = sample_dict.get(model)
                if score is not None:
                    avg_scores[model].append(score)

        # === Average score ===
        avg = {m: round(np.mean(scores), 2) if scores else None
               for m, scores in avg_scores.items()}

        # === Head-to-head win rate ===
        win_count = {m: 0 for m in model_list}       
        played_count = {m: 0 for m in model_list}     

        for sample_dict in samples.values():
            # List of (model, score) that are present AND allowed in this sample
            present = [(m, sample_dict[m]) for m in model_list if m in sample_dict]
            
            if len(present) < 2:
                continue

            max_score = max(s for _, s in present)
            winners = [m for m, s in present if s == max_score]

            if len(winners) == 1:
                winner = winners[0]
                win_count[winner] += 1
                for m, _ in present:
                    if m != winner:
                        played_count[m] += 1
                played_count[winner] += (len(present) - 1)
            else:
                for m, _ in present:
                    played_count[m] += (len(present) - 1)

        # Win rate % = (wins / games played) × 100
        winrate = {}
        for m in model_list: 
            games = played_count[m]
            wins = win_count[m]
            winrate[m] = round(wins / games * 100, 1) if games > 0 else 0.0

        metrics[task] = {
            "average": avg,
            "winrate_%": winrate,
            "samples": n,
            "scale_applied": scale if scale != 1.0 else "none (already ≥5)"
        }

    # Optionally re-save normalized files
    for task, data in normalized_data.items():
        # Only keep scores for allowed models in the output file
        filtered_data = {}
        for key, sample in data.items():
            filtered_data[key] = {m: s for m, s in sample.items() if m in allowed_models}
            
        out_path = os.path.join(OUTPUT_DIR, f"{task}_eval_normalized.json")
        with open(out_path, "w") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    return metrics, normalized_data

def plot_bar_comparison(metrics, allowed_models):
    tasks = list(metrics.keys())
    n_tasks = len(tasks)
    
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]
    
    model_list = sorted(list(allowed_models)) 
    
    for task, ax in zip(tasks, axes):
        avg = metrics[task]["average"]

        # Extract (model_key, score) pairs
        items = [(m, avg.get(m, 0) or 0) for m in model_list]

        # Sort descending by score
        items.sort(key=lambda x: x[1], reverse=True)

        # Unpack sorted items
        sorted_model_keys = [m for m, _ in items]
        sorted_scores = [s for _, s in items]
        
        # --- Use MODEL_MAP for display names ---
        sorted_display_names = [MODEL_MAP.get(m, m) for m in sorted_model_keys]
        sorted_colors = [MODEL_COLORS.get(m, "#333333") for m in sorted_model_keys]

        bars = ax.bar(sorted_display_names, sorted_scores, # Use display names for bar positions
                      color=sorted_colors, edgecolor='black', linewidth=0.8)

        ax.set_title(task.upper().replace("_", " "), fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticklabels(sorted_display_names, rotation=90, ha='center', fontsize=9) # Use display names for labels
        
        for bar, score in zip(bars, sorted_scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, score + 2, f"{score:.1f}",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.suptitle("Average Score per Task (normalized to 0–100 where needed)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task_comparison_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/task_comparison_bar.png")

def plot_radar_chart(metrics, allowed_models):
    tasks = list(metrics.keys())
    angles = [n / float(len(tasks)) * 2 * pi for n in range(len(tasks))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Use the allowed models list
    for model_key in allowed_models:
        values = []
        for task in tasks:
            avg = metrics[task]["average"].get(model_key)
            values.append(avg if avg is not None else 0) 
        values += values[:1]
        
        # --- Use MODEL_MAP for display names in the legend ---
        display_name = MODEL_MAP.get(model_key, model_key)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=display_name, 
                color=MODEL_COLORS.get(model_key, "#333333"))
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.upper().replace("_", " ") for t in tasks])
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title("Overall Performance Radar (Normalized 0–100)", fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_radar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/overall_radar_chart.png")

def save_summary_table(metrics, allowed_models):
    lines = ["# Model Performance Summary\n"]
    lines.append("*Normalized to 0–100 where max < 5*\n")
    
    # Build header
    header = "| Model |"
    separator = "|-------|"
    for task in metrics.keys():
        header += f" {task} |"
        separator += "---------|"
    header += " Avg Rank |"
    separator += "----------|"
    
    lines.append(header)
    lines.append(separator)
    
    # Compute average across tasks
    model_total = {m: 0 for m in allowed_models}
    model_count = {m: 0 for m in allowed_models}
    
    for task in metrics.keys():
        for m, score in metrics[task]["average"].items():
            if m in allowed_models and score is not None:
                model_total[m] += score
                model_count[m] += 1
    
    avg_across_tasks = {m: round(model_total[m]/model_count[m], 1) if model_count[m] else 0 for m in allowed_models}
    
    # Sort by overall average
    sorted_model_keys = sorted(list(allowed_models), key=lambda m: avg_across_tasks[m], reverse=True)
    
    for rank, model_key in enumerate(sorted_model_keys, 1):
        # --- Use MODEL_MAP for display name in the table ---
        display_name = MODEL_MAP.get(model_key, model_key)
        
        row = f"| **{display_name}** |"
        for task in metrics.keys():
            score = metrics[task]["average"].get(model_key)
            win = metrics[task]["winrate_%"].get(model_key, 0)
            if score is not None:
                row += f" {score:.1f} (↑{win}%) |"
            else:
                row += " — |"
        row += f" **#{rank}** |"
        lines.append(row)
    
    summary_md = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "SUMMARY.md"), "w") as f:
        f.write(summary_md)
    print(f"Saved: {OUTPUT_DIR}/SUMMARY.md")

# ================= RUN =================
if __name__ == "__main__":
    print("=" * 60)
    print("LOADING & COMPUTING METRICS")
    print("=" * 60)
    
    print("\nLoading task evaluation files from finetune/*.eval.json...")
    task_data, all_models_found = load_task_data()
    
    if not task_data:
        print("\nERROR: No task data loaded. Please check that *_task_eval.json files exist in finetune/ directory.")
        exit(1)
        
    # --- START OF FILTERING LOGIC ---
    print("\nFiltering models...")
    allowed_list = sorted(list(ALLOWED_MODELS.intersection(set(all_models_found))))
    print(allowed_list)
    print(f"Models allowed (from MODEL_COLORS): {sorted(list(ALLOWED_MODELS))}")
    print(f"Models used in analysis (found in files AND allowed): {allowed_list}")
    # --- END OF FILTERING LOGIC ---
    
    if not allowed_list:
        print("\nERROR: No allowed models found in the evaluation files. Check MODEL_COLORS keys.")
        exit(1)
    
    print("\nComputing metrics & normalizing...")
    metrics, _ = compute_metrics(task_data, allowed_list)
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_bar_comparison(metrics, allowed_list)
    plot_radar_chart(metrics, allowed_list)
    
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    save_summary_table(metrics, allowed_list)
    
    # Final JSON with all metrics
    with open(os.path.join(OUTPUT_DIR, "full_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/full_metrics.json")
    
    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)
    print(f"\nCheck the folder: {OUTPUT_DIR}/")
    print("   • task_comparison_bar.png")
    print("   • overall_radar_chart.png")
    print("   • SUMMARY.md")
    print("   • full_metrics.json")
    print("   • *_eval_normalized.json (normalized data)")