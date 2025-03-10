import pickle
import json
import random
from collections import Counter

# Load the pickle file
with open(
    "/BS/dduka/work/projects/TempNet/Bimodal_CL/pickle/caption_features_without_tensors.pkl",
    "rb",
) as f:
    data = pickle.load(f)
    counter = Counter(data["metadata"]["classes"]).most_common()
    print(counter)
    del data["metadata"]

with open(
    "/BS/dduka/work/projects/TempNet/Bimodal_CL/cc3m_extracted/merged.json", "r"
) as f:
    captions = json.load(f)
    captions = {k.split("_", 1)[1]: v for k, v in captions.items()}

x_class_entries = [key for key, entry in data.items() if entry.get("class_") == 104]

# Select 10 random entries (or less if there aren't 10 available)
num_to_select = min(200, len(x_class_entries))
random_selection = random.sample(x_class_entries, num_to_select)

# Print the selected entries
for i, entry in enumerate(random_selection, 1):
    print(f"Entry {i}:")
    print(captions[entry])
