import os
from collections import defaultdict, Counter
from PIL import Image
import json
from tqdm import tqdm
# Set your multiple base directories here
base_dirs = ['/home/user/kz-mm/data/DataSet_1', '/home/user/kz-mm/data/DataSet_2', '/home/user/kz-mm/data/DataSet_3', '/home/user/kz-mm/data/DataSet_4']

# Initialize stats
total_files = 0
file_type_counts = Counter()
resolution_counts = Counter()
resolution_groups = {'small': 0, 'medium': 0, 'large': 0}
cluster_image_counts = defaultdict(int)

# Group thresholds
def get_resolution_group(width, height):
    pixels = width * height
    if pixels < 500_000:
        return 'small'
    elif pixels <= 2_000_000:
        return 'medium'
    else:
        return 'large'

# Traverse all base_dirs
for base_dir in base_dirs:
    for cluster in tqdm(os.listdir(base_dir)):
        cluster_path = os.path.join(base_dir, cluster)
        if not os.path.isdir(cluster_path):
            continue

        for subcluster in tqdm(os.listdir(cluster_path)):
            subcluster_path = os.path.join(cluster_path, subcluster)
            if not os.path.isdir(subcluster_path):
                continue

            for topic in os.listdir(subcluster_path):
                topic_path = os.path.join(subcluster_path, topic)
                if not os.path.isdir(topic_path):
                    continue

                for fname in os.listdir(topic_path):
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        continue

                    fpath = os.path.join(topic_path, fname)
                    ext = fname.split('.')[-1].lower()
                    file_type_counts[ext] += 1
                    total_files += 1
                    cluster_image_counts[cluster] += 1

                    try:
                        with Image.open(fpath) as img:
                            width, height = img.size
                            res_str = f'{width}x{height}'
                            resolution_counts[res_str] += 1
                            group = get_resolution_group(width, height)
                            resolution_groups[group] += 1
                    except Exception as e:
                        print(f"Warning: could not read {fpath}: {e}")

# Compose summary
summary = {
    'total_files': total_files,
    'file_type_counts': dict(file_type_counts),
    'cluster_image_counts': dict(cluster_image_counts),
    'resolution_counts': dict(resolution_counts),
    'resolution_groups': resolution_groups
}

# Save to JSON
with open('datasets_image_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
# Pretty print with emojis
print("\n📊 📁 Dataset Image Statistics Summary 📁 📊")
print("=" * 60)
print(f"🖼️ Total Images Processed: {total_files:,}")
print("\n🧾 File Type Distribution:")
for ext, count in file_type_counts.items():
    print(f"  📂 .{ext.upper():<5} → {count:,}")

print("\n🧮 Cluster-wise Image Counts:")
for cluster, count in sorted(cluster_image_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  🧱 {cluster:<20} → {count:,} images")

print("\n📐 Resolution Grouping:")
for group, count in resolution_groups.items():
    emoji = "🔹" if group == "small" else ("🔸" if group == "medium" else "🔶")
    print(f"  {emoji} {group.capitalize():<7} → {count:,}")

print("\n📏 Top 5 Most Common Resolutions:")
top_resolutions = sorted(resolution_counts.items(), key=lambda x: x[1], reverse=True)[:5]
for res, count in top_resolutions:
    print(f"  🖼️ {res:<12} → {count:,} images")

print("=" * 60)
print("✅ Summary saved to `datasets_image_statistics.json`")