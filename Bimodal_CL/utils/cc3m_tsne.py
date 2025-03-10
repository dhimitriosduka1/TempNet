import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
import pickle
import torch
from collections import Counter, defaultdict
from tqdm import tqdm
import random


def plot_tsne_visualization(
    clip_embeddings, class_labels, perplexity=30, n_iter=1000, random_state=42
):
    reducer = TSNE(
        perplexity=perplexity,
        n_iter=n_iter,
        n_components=2,
        verbose=1,
        random_state=random_state,
    )

    print("Performing t-SNE...")
    tsne_embeddings = reducer.fit_transform(clip_embeddings)

    np.save("tsne_embeddings_v3.npy", tsne_embeddings)

    plt.figure(figsize=(16, 12), dpi=300)

    cmap = cm.jet
    norm = Normalize(vmin=0, vmax=199)

    print("Plotting...")
    scatter = plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=class_labels,
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.8,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label("Class ID", fontsize=28)

    plt.title("t-SNE Visualization of Text Embeddings", fontsize=32)
    plt.xlabel("t-SNE Dimension 1", fontsize=28)
    plt.ylabel("t-SNE Dimension 2", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"cc3m_tsne_v3.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(f"cc3m_tsne_v3.png", bbox_inches="tight", pad_inches=0.1)

    return tsne_embeddings


# Set random seed for reproducibility
random.seed(42)

# Load the classes of interest
with open(
    "/BS/dduka/work/projects/TempNet/Bimodal_CL/pickle/caption_features_without_tensors.pkl",
    "rb",
) as f:
    data = pickle.load(f)["metadata"]["classes"]
    data = Counter(data).items()
    data = sorted(data, key=lambda x: int(x[1]), reverse=True)
    classes_of_interest = [int(class_) for class_, _ in (data[:20] + data[-20:])]

# Load and subsample the features
with open("/BS/dduka/work/projects/TempNet/Bimodal_CL/caption_features.pkl", "rb") as f:
    data = pickle.load(f)

    # Group features by class
    features_by_class = defaultdict(list)

    print("Grouping features by class...")
    for key, data_item in tqdm(data.items()):
        if key == "metadata":
            continue

        feature = data_item["features"]
        class_ = int(data_item["class_"])

        if class_ not in classes_of_interest:
            continue

        features_by_class[class_].append(feature)

    # Subsample each class by a factor of 7
    features = []
    labels = []

    print("Subsampling classes by a factor of 7...")
    for class_, class_features in features_by_class.items():
        # Take every 7th sample
        subsampled_features = class_features[::7]

        print(
            f"Class {class_}: Original samples: {len(class_features)}, After subsampling: {len(subsampled_features)}"
        )

        features.extend(subsampled_features)
        labels.extend([class_] * len(subsampled_features))

    if features:
        features = torch.stack(features).numpy()

        print(f"Shape of features after subsampling: {features.shape}")
        print(f"Number of unique classes: {len(set(labels))}")

        tsne_result = plot_tsne_visualization(features, labels)
    else:
        print(
            "No features found after subsampling. Check if classes_of_interest contains valid classes."
        )
