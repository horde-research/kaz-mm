# 🖼️ Dataset Processing Tools

This repository contains a suite of scripts for managing, analyzing, and cleaning image dataset with a hierarchical structure (`cluster/subcluster/topic`). Below is a brief description of each script.

---

## 📊 Script Overview

### `aesthetic_scorer.py`
🔍 **Purpose**: Runs an aesthetic scoring model on all images in each topic folder.

- Uses CLIP and a linear aesthetic predictor to assign a score to each image.
- Saves results in `aesthetic_data.json` within each topic directory in the format:  
  `{"image1.jpg": 6.7, "image2.png": 4.3}`

---

### `aesthetic_stats.py`
📈 **Purpose**: Analyzes aesthetic scores across the dataset.

- Computes **median aesthetic score** per topic.
- For each subcluster, identifies:
  - Top K topics (by median aesthetic score)
  - Bottom K topics (by median aesthetic score)
- Includes emoji-rich summary output for better readability.

---

### `dataset_stats.py`
📷 **Purpose**: Generates statistics about image resolution and formats.

- Counts total images, file types, and resolutions.
- Groups resolutions into:
  - Small (<0.5 MP)
  - Medium (0.5–2 MP)
  - Large (>2 MP)
- Saves stats in `datasets_image_statistics.json`.
- Ends with a pretty-printed emoji summary.

---

### `duplicates_stats.py`
🧬 **Purpose**: Summarizes duplicate data across datasets.

- Reads `duplicates_*.json` files from all datasets.
- Calculates:
  - Total number of duplicate image entries.
  - Median number of images and duplicates per cluster.
- Pretty-prints stats for quick inspection.

---

### `find_duplicates.py`
🔁 **Purpose**: Detects duplicate images in a topic folder.

- Compares images by perceptual hash (pHash) or CLIP embedding similarity.
- Outputs a `duplicates_<topic>.json` file mapping each image to its duplicates.

---

### `merge_clusters.py`
📦 **Purpose**: Merges all clusters from multiple datasets into a single folder.

- Merges all clusters from multiple base directories into a new folder: `full_dataset/`.
- Skips copying duplicate images by checking `duplicates_<topic>.json` located in each **subcluster folder**.

---

### `split_data.py`
✂️ **Purpose**: Splits the dataset into train/validation sets.

- For each topic:
  1. Sorts images by resolution × aesthetic score (descending).
  2. Selects top 10 images as train, next 5 as validation.
- Outputs:
  - `train_set.json`
  - `valid_set.json`

---

## 📁 Folder Structure Example

```
DataSet_X/
└── ClusterA/
    └── Subcluster1/
        ├── TopicFoo/
        │   ├── image1.jpg
        │   └── aesthetic_data.json
        └── duplicates_TopicFoo.json
```

---

### `gpt_captioning_for_prompt_testing.py`
📝 **Purpose**: Creates a small test set of Kazakh captions for images
- Loads image paths from `train_set_Xsmall.json`
- Groups images by category/subcategory.
- Picks one random image per group (excluding the first).
- Sends each image to GPT with a strict 5-sentence factual captioning prompt.
- Output: `captions_output.json`

---

### `judge_captions.py`
⚖️ **Purpose**: Automatically chooses the better of two generated captions per image using GPT.
- Loads images and two captions from `captions_output.json`
- Sends each image and captions to OpenAI with a strict judging prompt.
- GPT returns "1" or "2" to select the better caption.
- Saves updated entries with the winner field into `judged_captions_output.json`

---


### `generate_file_map.py`
🔗 **Purpose**: Creates a JSON map linking local image paths to Hugging Face URLs.
- Scans all files in destination folder.
- For each file:
  - Builds a Hugging Face download URL based on its relative path.
  - Normalizes Unicode (NFD) and URL-encodes special characters.
- Output: `file_map.json`

---

### `batch_jsonl_prepare.py`
🗂️ **Purpose**: Prepares a .jsonl file for OpenAI batch captioning.
- Loads image paths from `valid_set_Xsmall.json` and URLs from `file_map.json`.
- For each image:
  - Creates a batch request with a detailed Kazakh captioning prompt.
  - Embeds the corresponding image URL.
- Output: `batch_input.jsonl`.

---

### `gpt_batch_run.py`
 ⚡ **Purpose**: Submits and monitors an OpenAI batch captioning job.
- Uploads `batch_input.jsonl` to OpenAI's batch API.
- Creates a batch job with completion window and monitors status.
- On success:
  - Downloads results. 
  - Saves to `output_valid_dataset_xsmall.json` for further processing.

---