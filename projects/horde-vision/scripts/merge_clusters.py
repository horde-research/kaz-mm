import os
import shutil
import json
from tqdm import tqdm

base_dirs = [
    '/home/user/kz-mm/data/DataSet_1',
    '/home/user/kz-mm/data/DataSet_2',
    '/home/user/kz-mm/data/DataSet_3',
    '/home/user/kz-mm/data/DataSet_4'
]

output_dir = '/home/user/kz-mm/data/full_dataset'
os.makedirs(output_dir, exist_ok=True)

for base_dir in base_dirs:
    for cluster in tqdm(os.listdir(base_dir), desc=f"Processing {base_dir}"):
        cluster_path = os.path.join(base_dir, cluster)
        if not os.path.isdir(cluster_path):
            continue

        for subcluster in os.listdir(cluster_path):
            subcluster_path = os.path.join(cluster_path, subcluster)
            if not os.path.isdir(subcluster_path):
                continue

            # Preload all duplicates in this subcluster once
            duplicates_by_topic = {}

            for fname in os.listdir(subcluster_path):
                if fname.startswith("duplicates_") and fname.endswith(".json"):
                    dup_json_path = os.path.join(subcluster_path, fname)
                    with open(dup_json_path, "r") as f:
                        dup_data = json.load(f)
                        duplicates = set()
                        for k, v in dup_data.items():
                            duplicates.update(v)
                        duplicates_by_topic[fname] = duplicates

            for topic in os.listdir(subcluster_path):
                topic_path = os.path.join(subcluster_path, topic)
                if not os.path.isdir(topic_path):
                    continue

                # Resolve the duplicate file
                dup_filename = f'duplicates_{topic}.json'
                topic_duplicates = duplicates_by_topic.get(dup_filename, set())

                # Destination path
                dest_topic_path = os.path.join(output_dir, cluster, subcluster, topic)
                os.makedirs(dest_topic_path, exist_ok=True)

                for fname in os.listdir(topic_path):
                    if fname in topic_duplicates:
                        continue  # Skip known duplicates

                    src_path = os.path.join(topic_path, fname)
                    dst_path = os.path.join(dest_topic_path, fname)

                    if os.path.isfile(src_path) and not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)