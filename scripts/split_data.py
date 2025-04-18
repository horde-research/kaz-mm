import os
import json
from PIL import Image
from tqdm import tqdm

base_dir = '/home/user/kz-mm/data/full_dataset'
train_samples = []
valid_samples = []

def compute_sorting_score(image_path, aesthetic_score):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return (width * height) * aesthetic_score
    except Exception as e:
        print(f"⚠️ Failed to open {image_path}: {e}")
        return -1

for cluster in tqdm(os.listdir(base_dir), desc="Clusters"):
    cluster_path = os.path.join(base_dir, cluster)
    if not os.path.isdir(cluster_path):
        continue

    for subcluster in os.listdir(cluster_path):
        subcluster_path = os.path.join(cluster_path, subcluster)
        if not os.path.isdir(subcluster_path):
            continue

        for topic in os.listdir(subcluster_path):
            topic_path = os.path.join(subcluster_path, topic)
            if not os.path.isdir(topic_path):
                continue

            json_path = os.path.join(topic_path, 'aesthetic_data.json')
            if not os.path.exists(json_path):
                continue

            with open(json_path) as f:
                aesthetics = json.load(f)

            image_scores = []
            for image_name, score in aesthetics.items():
                image_path = os.path.join(topic_path, image_name)
                try:
                    sort_score = compute_sorting_score(image_path, score)
                except:
                    continue
                if sort_score > 0:
                    image_scores.append((sort_score, image_path))

            image_scores.sort(reverse=True)

            train_samples.extend([img for _, img in image_scores[:4]])
            valid_samples.extend([img for _, img in image_scores[4:6]])

# Save to JSON
with open('train_set_Xsmall.json', 'w') as f:
    json.dump(train_samples, f, indent=2)

with open('valid_set_Xsmall.json', 'w') as f:
    json.dump(valid_samples, f, indent=2)

# Print summary
print(f"✅ Train set: {len(train_samples)} images")
print(f"✅ Valid set: {len(valid_samples)} images")