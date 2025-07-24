# \myparagraph{Modality Gap.}
# \begin{table}[h]
#     \centering
#     \resizebox{.95\columnwidth}{!}{
#         \begin{tabular}{ccccccc}
#             \textbf{Base} & \textbf{Mod.} & \textbf{U.L} & \textbf{Sch.} & MSCOCO & Flickr30K & CC3M \\
#             \midrule
#             \ding{52} & -- & -- & -- & 0.69 & 0.65 & 0.60 \\
#             \midrule
#             \ding{52} & \ding{52} & -- & \ding{52} & 0.70 & 0.65 & 0.63 \\
#             \ding{52} & \ding{52} & \ding{52} & \ding{52} & 0.44 & 0.39 & 0.28 \\
#         \end{tabular}
#     }
# \caption{\textbf{Modality-gap values for fully fine-tuned models.} Computed following the protocol of~\cite{liang2022mind}.}
# \label{tab:modality_gap_per_dataset}
# \end{table}

# \myparagraph{k-NN Evaluation.}
# \begin{table}[h]
#         \centering
#         \resizebox{.95\columnwidth}{!}{
#             \begin{tabular}{cccccccc}
#                 \toprule
#                 \multirow{2}{*}{\textbf{Base}} &
#                 \multirow{2}{*}{\textbf{Mod.}} &
#                 \multirow{2}{*}{\textbf{U.L}} &
#                 \multirow{2}{*}{\textbf{Sch.}} &
#                 \multicolumn{2}{c}{\textbf{CIFAR10}} &
#                 \multicolumn{2}{c}{\textbf{CIFAR100}} \\
#                  & & & & \textbf{k-NN@1} & \textbf{k-NN@10} & \textbf{k-NN@1} & \textbf{k-NN@10} \\
#                 \midrule
#                 \ding{52} & -- & -- & -- & 73.40 & 77.29 & 50.59 & 53.27\\
#                 \midrule
#                 \ding{52} & \ding{52} & -- & \ding{52} & 72.14 & 76.13 & 49.44 & 52.55 \\
#                 \ding{52} & \ding{52} & \ding{52} & \ding{52} & 80.70 & 83.89 & 57.04 & 60.65 \\
#             \end{tabular}
#         }
# \caption{\textbf{k-NN Performance in the Image Modality.}
# Accuracy of fully fine-tuned models evaluated with k-NN classification.}
# \label{tab:knn_evaluation}
# \end{table}

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots
import os


# Modality Gap
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


COL_W_IN = 3.487
ASPECT = 0.62

# ---------- Bigger fonts ----------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.size": 9,  # base
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def plot_modality_gap(
    data,
    output_dir=".",
    filename="/BS/dduka/work/projects/TempNet/Bimodal_CL/knn_and_modality_gap_visualization/modality_gap_visualization.pdf",
    save_plots=True,
):
    plt.style.use(["science", "no-latex"])

    methods = list(data.keys())
    datasets = list(data[methods[0]].keys())
    x = np.arange(len(datasets))
    width = 0.25
    positions = [x - width, x, x + width]

    method_colors = {
        "InfoNCE": "#FF6B6B",
        "InfoNCE + Mod. InfoNCE": "#4ECDC4",
        "TeMo": "#45B7D1",
    }

    fig, ax = plt.subplots(
        figsize=(COL_W_IN, COL_W_IN * ASPECT), constrained_layout=False
    )

    # bars
    for i, m in enumerate(methods):
        vals = [data[m][d] for d in datasets]
        ax.bar(
            positions[i],
            vals,
            width,
            label=m,
            color=method_colors.get(m, f"C{i}"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )

    # axes
    ax.set_title("Modality Gap Comparison", pad=6, fontsize=14)

    ax.set_xlabel("Datasets")
    ax.set_ylabel("Modality Gap")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)

    ax.grid(True, axis="y", color="#E0E0E0", alpha=0.4, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(v.values()) for v in data.values()) * 1.15)

    # legend BELOW plot with padding
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),  # more space
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.6,
    )

    # reserve space at bottom
    fig.subplots_adjust(bottom=0.20)

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
        print(f"Saved: {out_path}")

    return fig


def load_knn_data():
    return {
        "InfoNCE": {
            "CIFAR10": {
                "k-NN@1": 73.40,
                "k-NN@10": 77.29,
            },
            "CIFAR100": {
                "k-NN@1": 50.59,
                "k-NN@10": 53.27,
            },
        },
        "InfoNCE + Mod. InfoNCE": {
            "CIFAR10": {
                "k-NN@1": 72.14,
                "k-NN@10": 76.13,
            },
            "CIFAR100": {
                "k-NN@1": 49.44,
                "k-NN@10": 52.55,
            },
        },
        "TeMo": {
            "CIFAR10": {
                "k-NN@1": 80.70,
                "k-NN@10": 83.89,
            },
            "CIFAR100": {
                "k-NN@1": 57.04,
                "k-NN@10": 60.65,
            },
        },
    }


def plot_knn1_single(
    data,
    output_dir=".",
    filename="knn1_visualization.pdf",
    save_plots=True,
):
    """
    One figure showing k-NN@1 for all methods on all datasets.

    Parameters
    ----------
    data : dict
        {method: {dataset: {"k-NN@1": val, "k-NN@10": val}}}
    output_dir : str
        Directory where the plot will be saved.
    filename : str
        Name of the output pdf file.
    save_plots : bool
        Whether to save the plot to disk.
    """
    # --- STYLE ---
    plt.style.use(["science", "no-latex"])

    methods = list(data.keys())
    datasets = list(next(iter(data.values())).keys())
    x = np.arange(len(datasets))
    width = 0.25
    positions = [
        x - width,
        x,
        x + width,
    ]  # assumes exactly 3 methods; generalize if needed

    method_colors = {
        "InfoNCE": "#FF6B6B",
        "InfoNCE + Mod. InfoNCE": "#4ECDC4",
        "TeMo": "#45B7D1",
    }

    fig, ax = plt.subplots(
        figsize=(COL_W_IN, COL_W_IN * ASPECT), constrained_layout=False
    )

    # --- BARS ---
    for i, m in enumerate(methods):
        vals = [data[m][ds]["k-NN@1"] for ds in datasets]
        ax.bar(
            positions[i],
            vals,
            width,
            label=m,
            color=method_colors.get(m, f"C{i}"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )

    # --- AXES ---
    ax.set_title("k-NN@1 Accuracy Comparison", pad=6, fontsize=14)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)

    ax.grid(True, axis="y", color="#E0E0E0", alpha=0.4, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = max(data[m][ds]["k-NN@1"] for m in methods for ds in datasets)
    ax.set_ylim(0, max_val * 1.15)

    # --- LEGEND BELOW PLOT ---
    handles, labels = ax.get_legend_handles_labels()
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
    )

    # Reserve space at bottom for legend
    fig.subplots_adjust(bottom=0.20)

    # --- SAVE ---
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
        print(f"Saved: {out_path}")

    return fig


def plot_knn1_line(
    data,
    output_dir="/BS/dduka/work/projects/TempNet/Bimodal_CL/knn_and_modality_gap_visualization",
    save_plots=True,
):
    """
    Line plot of k-NN@1:
      - X-axis: methods
      - One line per dataset (legend = datasets)
      - No star markers
      - Y-axis fixed to 0-100
    """
    # ---- STYLE ----
    plt.style.use(["science", "no-latex"])
    colors = {"background": "#FAFAFA", "grid": "#E0E0E0", "text": "#2C3E50"}

    dataset_colors = {"CIFAR10": "#4E79A7", "CIFAR100": "#F28E2B"}

    methods = list(data.keys())
    datasets = list(next(iter(data.values())).keys())
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor("white")

    # Plot each dataset
    for ds in datasets:
        y = [data[m][ds]["k-NN@1"] for m in methods]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label=ds,
            color=dataset_colors.get(ds, None),
        )
        # annotate values
        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + 0.3,
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=colors["text"],
            )

    # Axes formatting
    ax.set_title(
        "k-NN@1 Accuracy (Methods on X, Datasets in Legend)",
        fontsize=20,
        fontweight="bold",
        color=colors["text"],
        pad=25,
    )
    ax.set_xlabel("Methods", fontsize=16, color=colors["text"])
    ax.set_ylabel("Accuracy (%)", fontsize=16, color=colors["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=15, color=colors["text"])

    ax.grid(True, alpha=0.3, color=colors["grid"], linewidth=0.5, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(colors["text"])
    ax.spines["bottom"].set_color(colors["text"])
    ax.tick_params(colors=colors["text"], which="both", labelsize=14)

    # Legend top-right
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=14,
        title="Datasets",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(colors["grid"])

    # Fixed y-axis
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "knn1_visualization.pdf")

        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=colors["background"],
            edgecolor="none",
        )
        print(f"Saved: {out_path}")

    return fig


def plot_modgap_and_knn1(
    modgap_data,
    knn_data,
    output_dir=".",
    filename="modgap_knn1_combined.pdf",
    save_plots=True,
):
    """
    Create a 2-panel figure:
      Left  : Modality Gap bars
      Right : k-NN@1 bars
    Both share a single legend at the bottom.

    Parameters
    ----------
    modgap_data : dict
        {method: {dataset: value}}
    knn_data : dict
        {method: {dataset: {"k-NN@1": val, "k-NN@10": val}}}
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename (PDF).
    save_plots : bool
        If True, save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    """
    # ----- STYLE -----
    plt.style.use(["science", "no-latex"])

    methods = list(modgap_data.keys())
    datasets_gap = list(next(iter(modgap_data.values())).keys())
    datasets_knn = list(next(iter(knn_data.values())).keys())

    # Generalized positioning for any number of methods
    width = 0.25

    def bar_positions(n_groups, n_methods, width):
        x = np.arange(n_groups)
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width
        return x, offsets

    method_colors = {
        "InfoNCE": "#FF6B6B",
        "InfoNCE + Mod. InfoNCE": "#4ECDC4",
        "TeMo": "#45B7D1",
    }

    # Figure size: two columns wide, keep same height ratio as before
    fig_w = COL_W_IN * 2.05  # small padding
    fig_h = COL_W_IN * ASPECT
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=False)
    ax_gap, ax_knn = axes

    # --------- PANEL 1: Modality Gap ---------
    x_gap, offsets_gap = bar_positions(len(datasets_gap), len(methods), width)
    for i, m in enumerate(methods):
        vals = [modgap_data[m][d] for d in datasets_gap]
        ax_gap.bar(
            x_gap + offsets_gap[i],
            vals,
            width,
            label=m,
            color=method_colors.get(m, f"C{i}"),
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

    # --------- PANEL 2: k-NN@1 ---------
    x_knn, offsets_knn = bar_positions(len(datasets_knn), len(methods), width)
    for i, m in enumerate(methods):
        vals = [knn_data[m][ds]["k-NN@1"] for ds in datasets_knn]
        ax_knn.bar(
            x_knn + offsets_knn[i],
            vals,
            width,
            label=m,  # same labels, we'll only keep one legend
            color=method_colors.get(m, f"C{i}"),
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
    ax_knn.set_ylim(0, max_knn * 1.15)

    # --------- ONE LEGEND (bottom) ---------
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

    # Reserve space for legend
    fig.subplots_adjust(bottom=0.22, wspace=0.35)

    # --------- SAVE ---------
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
        print(f"Saved: {out_path}")

    return fig, axes


# Load modality gap data
modality_gap_data = load_modality_gap_data()
# plot_modality_gap(modality_gap_data)

# Load k-NN data
knn_data = load_knn_data()
# plot_knn1_single(knn_data)

plot_modgap_and_knn1(modality_gap_data, knn_data)