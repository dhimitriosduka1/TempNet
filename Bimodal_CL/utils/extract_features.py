from collections import Counter
import torch
import pickle
import open_clip
import json
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai"
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

# Load captions file
with open(
    "/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/merged.json", "r"
) as f:
    captions = json.load(f)

batch_size = 2048 * 16

# Prepare to store caption features
caption_features = {}
text_features = []

# Encode captions in batches
with torch.no_grad(), torch.cuda.amp.autocast():
    # Extract filenames and captions
    all_filenames = list(captions.keys())
    all_caption_texts = list(captions.values())

    # Process captions in batches
    for i in tqdm(range(0, len(all_caption_texts), batch_size)):
        batch_filenames = all_filenames[i : i + batch_size]
        batch_captions = all_caption_texts[i : i + batch_size]

        # Tokenize the batch of captions
        text_tokens = tokenizer(batch_captions).to(device)

        # Encode text features for the batch
        batch_text_features = model.encode_text(text_tokens)
        batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)

        # Store features with filename as key
        for filename, text_feature in zip(batch_filenames, batch_text_features):
            # taridx_sampleid -> sampleid
            filename = filename.split("_", 1)[1]

            caption_features[filename] = {}
            caption_features[filename]["features"] = text_feature.cpu()
            text_features.append(text_feature.cpu().numpy())

text_features = np.array(text_features)
print("Text features shape:", text_features.shape)

print("Performing KMeans on text features")
kmeans = KMeans(n_clusters=200, random_state=0)
kmeans.fit(text_features)
labels = kmeans.labels_

for i, (filename, data) in enumerate(caption_features.items()):
    caption_features[filename]["class_"] = labels[i]

print(f"Keys length: {len(captions)}, labels length: {len(labels)}")

# Append some metadata
counter = Counter([l for l in labels])
caption_features["metadata"] = {"classes": {}}
for class_, count in counter.items():
    caption_features["metadata"]["classes"][class_] = count

try:
    with open("caption_features_without_tensors.pkl", "wb") as f:
        pickle.dump(caption_features, f)
    print("Caption features saved successfully.")
    print(f"Total captions processed: {len(caption_features)}")
except Exception as e:
    print(f"Error saving features: {e}")
