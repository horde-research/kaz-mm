import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

base_dirs = [
    '/home/user/kz-mm/data/DataSet_1',
    '/home/user/kz-mm/data/DataSet_2',
    '/home/user/kz-mm/data/DataSet_3',
    '/home/user/kz-mm/data/DataSet_4'
]

subcluster_scores = defaultdict(list)  # {subcluster_path: [topic medians]}
topic_medians = {}  # optional: store per-topic stats if you want

def compute_median_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            scores = list(data.values())
            if scores:
                return float(np.median(scores))
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
    return None

# Traverse and collect medians
for base_dir in base_dirs:
    print(f"\n📁 Starting in base folder: {base_dir}")
    
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

                json_path = os.path.join(topic_path, "aesthetic_data.json")

                try:
                    median = compute_median_from_json(json_path)
                    if median is not None:
                        subcluster_scores[subcluster_path].append(median)
                        topic_medians[topic_path] = median
                except:
                    print(f"no data: {topic_path}")

print("\n✅ Aggregation complete!")

# Compute subcluster medians
subcluster_medians = {
    sub: float(np.median(scores)) for sub, scores in subcluster_scores.items() if scores
}
n_candidates = 5
# Sort
sorted_subclusters = sorted(subcluster_medians.items(), key=lambda x: x[1], reverse=True)
top_ = sorted_subclusters[:n_candidates]
bottom_ = sorted_subclusters[-n_candidates:]

# Print result
print(f"\n🏆 Top {n_candidates} subclusters by median aesthetic score:")
for path, median in top_:
    path_name = f"{path.split('/')[-2]}-{path.split('/')[-1]}"
    print(f"{path_name} → median: {median:.3f}")

print(f"\n📉 Bottom {n_candidates} subclusters by median aesthetic score:")
for path, median in bottom_:
    path_name = f"{path.split('/')[-2]}-{path.split('/')[-1]}"
    print(f"{path_name} → median: {median:.3f}")