import pickle
import matplotlib.pyplot as plt

from collections import Counter

cc3m_distribution = pickle.load(
    open(
        "/BS/dduka/work/projects/TempNet/Bimodal_CL/caption_features_without_tensors.pkl",
        "rb",
    )
)["metadata"]["classes"]

data = Counter(cc3m_distribution).items()
data = sorted(data, key=lambda x: int(x[1]), reverse=True)

template_counts = [item[1] for item in data]

# plot the distribution of the templates
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(range(len(template_counts)), template_counts)
plt.fill_between(
    range(len(template_counts)), template_counts, color="skyblue", alpha=0.4
)
plt.xlabel("Cluster Labels")
plt.ylabel("# of Samples")
plt.title(f"CC3M Cluster Size Distribution")
plt.savefig(f"cc3m_class_distribution.pdf")
