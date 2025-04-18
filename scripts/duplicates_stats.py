import os
import json
import numpy as np

summary_files = [
    'duplicates_summary_DataSet_1.json',
    'duplicates_summary_DataSet_2.json',
    'duplicates_summary_DataSet_3.json',
    'duplicates_summary_DataSet_4.json',
]

total_duplicates = 0
all_cluster_total_images = []
all_cluster_duplicate_images = []

for file in summary_files:
    print(f"🔍 Reading: {file}")
    try:
        with open(file, 'r') as f:
            data = json.load(f)

        total_duplicates += data.get('total_duplicate_images', 0)

        for cluster, stats in data.get('cluster_stats', {}).items():
            all_cluster_total_images.append(stats.get('total_images', 0))
            all_cluster_duplicate_images.append(stats.get('duplicate_images', 0))

    except Exception as e:
        print(f"❌ Failed to read {file}: {e}")

# Calculate medians
median_total_images = float(np.median(all_cluster_total_images)) if all_cluster_total_images else 0
median_duplicate_images = float(np.median(all_cluster_duplicate_images)) if all_cluster_duplicate_images else 0

# Output results
print("\n📊 Duplicates Aggregated Stats:")
print(f"🧮 Total duplicate images across all clusters: {total_duplicates}")
print(f"📐 Median total images per cluster: {int(median_total_images)}")
print(f"📐 Median duplicate images per cluster: {int(median_duplicate_images)}")