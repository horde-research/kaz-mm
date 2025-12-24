import os
import json
import torch
import torch.nn as nn
from PIL import Image
from urllib.request import urlretrieve
from os.path import expanduser
from tqdm import tqdm
import open_clip

# ----------------------- Model Loading Helpers -----------------------
def get_aesthetic_model(clip_model="vit_l_14"):
    """Load the aesthetic model for the given clip_model variant."""
    home = expanduser("~")
    cache_folder = os.path.join(home, ".cache", "emb_reader")
    os.makedirs(cache_folder, exist_ok=True)
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(path_to_model):
        # Download the aesthetic model weights.
        url_model = (
            f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        )
        print(f"Downloading aesthetic model to {path_to_model}")
        urlretrieve(url_model, path_to_model)
    # Build the linear mapping layer according to the chosen CLIP model.
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError(f"Unknown clip model: {clip_model}")
    s = torch.load(path_to_model, map_location='cpu')
    m.load_state_dict(s)
    m.eval()
    return m

# ----------------------- Set Device and Load Models -----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load aesthetic model
amodel = get_aesthetic_model(clip_model="vit_l_14").to(device)
amodel.eval()

# Load CLIP model and transforms from open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.to(device)
model.eval()

# ----------------------- Processing a Topic -----------------------
def process_topic(topic_dir, batch_size=32):
    """
    Run batched aesthetic model inference on all images in topic_dir.
    Save results into aesthetic_data.json.
    """
    aesthetic_scores = {}
    image_files = sorted([f for f in os.listdir(topic_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    images_to_process = []
    image_names = []

    for img_file in tqdm(image_files, desc=f"Loading {os.path.basename(topic_dir)}", leave=False):
        image_path = os.path.join(topic_dir, img_file)
        try:
            image = Image.open(image_path).convert("RGB")
            preprocessed = preprocess(image)
            images_to_process.append(preprocessed)
            image_names.append(img_file)
        except Exception as e:
            print(f"Warning: Skipping {image_path}. Error: {e}")

    if not images_to_process:
        print(f"No valid images in {topic_dir}")
        return {}

    # Batching
    with torch.no_grad():
        for i in range(0, len(images_to_process), batch_size):
            batch = images_to_process[i:i+batch_size]
            names = image_names[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(device)

            # Encode with CLIP
            image_features = model.encode_image(batch_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Predict aesthetic scores
            scores = amodel(image_features).squeeze(1).tolist()

            for name, score in zip(names, scores):
                aesthetic_scores[name] = float(score)

    # Save to JSON
    output_path = os.path.join(topic_dir, "aesthetic_data.json")
    with open(output_path, 'w') as f:
        json.dump(aesthetic_scores, f, indent=2)
    print(f"Saved {len(aesthetic_scores)} scores to {output_path}")
    return aesthetic_scores

# ----------------------- Directory Traversal -----------------------
base_dirs = ['/home/user/kz-mm/data/DataSet_1','/home/user/kz-mm/data/DataSet_2','/home/user/kz-mm/data/DataSet_3','/home/user/kz-mm/data/DataSet_4']

for base_dir in base_dirs:

    print(f"Starting aesthetic inference in base folder: {base_dir}")
    
    # Traverse clusters -> subclusters -> topics
    for cluster in tqdm(os.listdir(base_dir), desc="Clusters"):
        print(f"\t{cluster}...")
        cluster_path = os.path.join(base_dir, cluster)
        if not os.path.isdir(cluster_path):
            continue
        for subcluster in os.listdir(cluster_path):
            print(f"\t{subcluster}...")
            subcluster_path = os.path.join(cluster_path, subcluster)
            if not os.path.isdir(subcluster_path):
                continue
            for topic in os.listdir(subcluster_path):
                topic_path = os.path.join(subcluster_path, topic)
                if not os.path.isdir(topic_path):
                    continue
                # Run aesthetic inference on each topic folder.
                process_topic(topic_path)
    
    print("Aesthetic inference complete!")