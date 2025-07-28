import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

# Optional: only import if you actually have these installed
try:
    import scienceplots  # noqa: F401
    HAVE_SCIENCEPLOTS = True
except ImportError:  # gracefully degrade if not installed
    HAVE_SCIENCEPLOTS = False

"""
This script reproduces all figures and **starts the k-NN axis at 45** while also
showing fewer y-axis tick labels so the remaining values render larger and cleaner.
You can control how many ticks appear via the `MAX_YTICKS` constant or the
`limit_yticks(ax, n)` helper, and the lower y-limit for the k-NN panel via the
`KNN_YMIN` constant or the `knn_ymin` parameter of `plot_modgap_and_knn1`.
"""

# -------------------- GLOBAL CONFIG --------------------
COL_W_IN = 3.487  # single-column width (inches) – change if your template differs
ASPECT = 0.62  # height/width ratio
MAX_YTICKS = 4  # <<<< fewer ticks here
KNN_YMIN = 45   # <<<< start k-NN axis at 45 by default

# ---------- Matplotlib rcParams (fonts, etc.) ----------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.size": 9,  # base
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }
)

# If scienceplots is available, use it for styling; otherwise skip
if HAVE_SCIENCEPLOTS:
    plt.style.use(["science", "no-latex"])

# -------------------- DATA LOADERS --------------------

def load_modality_gap_data():
    return {
        "InfoNCE": {
            "MSCOCO": 0.69,
            "Flickr30K": 0.65,
            "CC3M": 0.60,
        },
        "InfoNCE + Mod. InfoNCE": {
            "MSCOCO": 0.70,
            "Flickr30K": 0.65,
            "CC3M": 0.63,
        },
        "TeMo": {
            "MSCOCO": 0.44,
            "Flickr30K": 0.39,
            "CC3M": 0.28,
        },
    }


def load_knn_data():
    return {
        "InfoNCE": {
            "CIFAR10": {"k-NN@1": 73.40, "k-NN@10": 77.29},
            "CIFAR100": {"k-NN@1": 50.59, "k-NN@10": 53.27},
        },
        "InfoNCE + Mod. InfoNCE": {
            "CIFAR10": {"k-NN@1": 72.14, "k-NN@10": 76.13},
            "CIFAR100": {"k-NN@1": 49.44, "k-NN@10": 52.55},
        },
        "TeMo": {
            "CIFAR10": {"k-NN@1": 80.70, "k-NN@10": 83.89},
            "CIFAR100": {"k-NN@1": 57.04, "k-NN@10": 60.65},
        },
    }


# -------------------- HELPERS --------------------

def limit_yticks(ax, n=MAX_YTICKS):
    """Limit the number of major y ticks to `n` and enlarge their labels.
    Call AFTER setting y-lims and plotting.
    """
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, prune=None))
    ax.tick_params(axis="y", labelsize=12)


def bar_positions(n_groups, n_methods, width):
    x = np.arange(n_groups)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width
    return x, offsets


# Consistent colors for methods
METHOD_COLORS = {
    "InfoNCE": "#FF6B6B",
    "InfoNCE + Mod. InfoNCE": "#4ECDC4",
    "TeMo": "#45B7D1",
}


# -------------------- PLOTS --------------------
def plot_modgap_and_knn1(
    modgap_data,
    knn_data,
    output_dir=".",
    filename="modgap_knn1_combined.pdf",
    save_plots=True,
    knn_ymin=KNN_YMIN,
):
    """
    2-panel figure:
      Left  : Modality Gap bars
      Right : k-NN@1 bars
    One legend at the bottom.

    Parameters
    ----------
    modgap_data : dict
        Output from `load_modality_gap_data()` or similar structure.
    knn_data : dict
        Output from `load_knn_data()` or similar structure.
    output_dir : str
        Directory to save the figure.
    filename : str
        Name of the saved PDF.
    save_plots : bool
        Whether to save the figure to disk.
    knn_ymin : float
        Lower bound for the k-NN@1 y-axis. Default is 45.
    """

    methods = list(modgap_data.keys())
    datasets_gap = list(next(iter(modgap_data.values())).keys())
    datasets_knn = list(next(iter(knn_data.values())).keys())

    width = 0.25
    fig_w = COL_W_IN * 2.05
    fig_h = COL_W_IN * ASPECT
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=False)
    ax_gap, ax_knn = axes

    # --- Panel 1: Modality Gap ---
    x_gap, offsets_gap = bar_positions(len(datasets_gap), len(methods), width)
    for i, m in enumerate(methods):
        vals = [modgap_data[m][d] for d in datasets_gap]
        ax_gap.bar(
            x_gap + offsets_gap[i],
            vals,
            width,
            label=m,
            color=METHOD_COLORS.get(m, f"C{i}"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
    ax_gap.set_title("Modality Gap", pad=6, fontsize=14)
    ax_gap.set_xlabel("Datasets")
    ax_gap.set_ylabel("Modality Gap")
    ax_gap.set_xticks(x_gap)
    ax_gap.set_xticklabels(datasets_gap)
    ax_gap.grid(True, axis="y", color="#E0E0E0", alpha=0.4, linewidth=0.4)
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)
    ax_gap.set_ylim(0, max(max(v.values()) for v in modgap_data.values()) * 1.15)

    # limit Y ticks here
    limit_yticks(ax_gap)

    # --- Panel 2: k-NN@1 ---
    x_knn, offsets_knn = bar_positions(len(datasets_knn), len(methods), width)
    for i, m in enumerate(methods):
        vals = [knn_data[m][ds]["k-NN@1"] for ds in datasets_knn]
        ax_knn.bar(
            x_knn + offsets_knn[i],
            vals,
            width,
            label=m,
            color=METHOD_COLORS.get(m, f"C{i}"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
    ax_knn.set_title("k-NN@1", pad=6, fontsize=14)
    ax_knn.set_xlabel("Datasets")
    ax_knn.set_ylabel("Accuracy (%)")
    ax_knn.set_xticks(x_knn)
    ax_knn.set_xticklabels(datasets_knn)
    ax_knn.grid(True, axis="y", color="#E0E0E0", alpha=0.4, linewidth=0.4)
    ax_knn.spines["top"].set_visible(False)
    ax_knn.spines["right"].set_visible(False)
    max_knn = max(knn_data[m][ds]["k-NN@1"] for m in methods for ds in datasets_knn)
    ax_knn.set_ylim(knn_ymin, max_knn * 1.15)

    # limit Y ticks here
    limit_yticks(ax_knn)

    # --- One legend ---
    handles, labels = ax_gap.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.6,
        fontsize=14,
    )

    fig.subplots_adjust(bottom=0.22, wspace=0.35)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
        print(f"Saved: {out_path}")

    return fig, axes


# -------------------- MAIN (example run) --------------------
if __name__ == "__main__":
    modality_gap_data = load_modality_gap_data()
    knn_data = load_knn_data()

    # Generate and save all figures
    plot_modgap_and_knn1(modality_gap_data, knn_data)
