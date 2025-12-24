import os
import json
from imagededup.methods import DHash
from collections import defaultdict
from tqdm import tqdm
import logging
logging.getLogger('imagededup').setLevel(logging.CRITICAL)
#base_dir = '/home/user/kz-mm/data/DataSet_1'

dhasher = DHash()

base_dirs = ['/home/user/kz-mm/data/DataSet_2','/home/user/kz-mm/data/DataSet_3','/home/user/kz-mm/data/DataSet_4']


for base_dir in base_dirs:
    curr_cluster_num = 1
    all_duplicates = {}
    total_images = 0
    total_duplicates = set()
    cluster_stats = {}
    num_clusters = len(os.listdir(base_dir))
    dataset_name = base_dir.split('/')[-1]
    print(base_dir)
    for cluster in tqdm(os.listdir(base_dir)):
        cluster_path = os.path.join(base_dir, cluster)
        if not os.path.isdir(cluster_path):
            continue
        print(f"Processing # {curr_cluster_num}: {cluster}")
        cluster_duplicates = {}
        cluster_images = 0
        cluster_dup_set = set()
    
        for subcluster in tqdm(os.listdir(cluster_path)):
            subcluster_path = os.path.join(cluster_path, subcluster)
            if not os.path.isdir(subcluster_path):
                continue
            print(f"\t{subcluster}...")
            for topic in os.listdir(subcluster_path):
                topic_path = os.path.join(subcluster_path, topic)
                if not os.path.isdir(topic_path):
                    continue
                print(f"\t\t{topic}...")
                try:
                    encodings = dhasher.encode_images(image_dir=topic_path)
                    duplicates = dhasher.find_duplicates(encoding_map=encodings)
                except Exception as e:
                    print(f"error!!! topic {topic_path} | {e}")
                    continue
    
                # Save per-topic duplicates
                output_path = os.path.join(subcluster_path, f'duplicates_{topic}.json')
                with open(output_path, 'w') as f:
                    json.dump(duplicates, f, indent=2)
    
                cluster_duplicates.update({
                    os.path.join(topic_path, k): [os.path.join(topic_path, dup) for dup in v]
                    for k, v in duplicates.items() if v
                })
    
                # Update stats
                cluster_images += len(encodings)
                for k, v in duplicates.items():
                    for dup in v:
                        cluster_dup_set.add(os.path.join(topic_path, dup))
                        cluster_dup_set.add(os.path.join(topic_path, k))
                        
        curr_cluster_num += 1
        all_duplicates[cluster] = cluster_duplicates
        total_images += cluster_images
        total_duplicates.update(cluster_dup_set)
        cluster_stats[cluster] = {
            'total_images': cluster_images,
            'duplicate_images': len(cluster_dup_set)
        }
    
    # Save global duplicates
    with open(f'all_duplicates_{dataset_name}.json', 'w') as f:
        json.dump(all_duplicates, f, indent=2)
    
    # Save summary
    summary = {
        'total_images': total_images,
        'total_duplicate_images': len(total_duplicates),
        'cluster_stats': cluster_stats
    }
    with open(f'duplicates_summary_{dataset_name}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print stats
    print(json.dumps(summary, indent=2))