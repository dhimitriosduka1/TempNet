import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import scienceplots

# -----------------------------------------------------------------------------
# Plotting style --------------------------------------------------------------
plt.style.use(["science"])
sns.set_palette("husl")

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

colors = {
    "start": "#FF6B6B",  # Coral red – start of training
    "middle": "#1f77b4",  # Blue       – end (InfoNCE)
    "end": "#4ECDC4",  # Teal       – end (TeMo)
    "start_dark": "#E55555",
    "middle_dark": "#1c6699",
    "end_dark": "#45B7AA",
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}


# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------
def load_pickle(path: str):
    """Load a pickle file and return its contents."""
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Data ------------------------------------------------------------------------
base_dir = "/BS/dduka/work/projects/TempNet/Bimodal_CL/visualizations/temperature_assignments_values"

datasets = {
    "sot": load_pickle(
        os.path.join(base_dir, "common/cc3m_temperature_assignments.pkl")
    ),
    "eot_infonce": load_pickle(
        os.path.join(base_dir, "infonce/cc3m_temperature_assignments.pkl")
    ),
    "eot_temo": load_pickle(
        os.path.join(base_dir, "temo/cc3m_temperature_assignments.pkl")
    ),
}

# -----------------------------------------------------------------------------
# Template describing what to plot --------------------------------------------
pairs = [
    {
        "title": "Positive Pairs Similarity Distribution",
        "tag": "i2t_positive",
        "series": [
            {
                "label": "Start of training",
                "values": datasets["sot"]["i2t_similarity_matrix_positive_pairs"]
                .cpu()
                .numpy(),
                "color": colors["start"],
                "dark_color": colors["start_dark"],
            },
            {
                "label": "End of training (InfoNCE)",
                "values": datasets["eot_infonce"][
                    "i2t_similarity_matrix_positive_pairs"
                ]
                .cpu()
                .numpy(),
                "color": colors["middle"],
                "dark_color": colors["middle_dark"],
            },
            {
                "label": "End of training (TeMo)",
                "values": datasets["eot_temo"]["i2t_similarity_matrix_positive_pairs"]
                .cpu()
                .numpy(),
                "color": colors["end"],
                "dark_color": colors["end_dark"],
            },
        ],
    },
    {
        "title": "Negative Pairs Similarity Distribution",
        "tag": "i2t_negative",
        "series": [
            {
                "label": "Start of training",
                "values": datasets["sot"]["i2t_similarity_matrix_negative_pairs"]
                .cpu()
                .numpy(),
                "color": colors["start"],
                "dark_color": colors["start_dark"],
            },
            {
                "label": "End of training (InfoNCE)",
                "values": datasets["eot_infonce"][
                    "i2t_similarity_matrix_negative_pairs"
                ]
                .cpu()
                .numpy(),
                "color": colors["middle"],
                "dark_color": colors["middle_dark"],
            },
            {
                "label": "End of training (TeMo)",
                "values": datasets["eot_temo"]["i2t_similarity_matrix_negative_pairs"]
                .cpu()
                .numpy(),
                "color": colors["end"],
                "dark_color": colors["end_dark"],
            },
        ],
    },
]

# -----------------------------------------------------------------------------
# Plotting --------------------------------------------------------------------
output_dir = "/BS/dduka/work/projects/TempNet/Bimodal_CL/temperature_assignments_values"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

n_bins = 5
alpha_fill = 0.7
alpha_line = 0.9


def create_histogram(title: str, tag: str, series: list):
    """Draw and save a smoothed histogram for multiple series."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor("white")

    data_min, data_max = -1.0, 1.0
    bins = np.linspace(data_min, data_max, n_bins)

    # -------------------------------------------------------------------------
    # Plot each series
    # -------------------------------------------------------------------------
    for s in series:
        vals = s["values"]
        kde = gaussian_kde(vals)
        x_smooth = np.linspace(data_min, data_max, 400)
        y_smooth = kde(x_smooth)

        ax.fill_between(
            x_smooth,
            y_smooth,
            alpha=alpha_fill,
            color=s["color"],
            label=s["label"],  # keep label for legend handles
            linewidth=0,
        )
        ax.plot(
            x_smooth,
            y_smooth,
            color=s["dark_color"],
            linewidth=2.5,
            alpha=alpha_line,
        )

    # -------------------------------------------------------------------------
    # Axes styling
    # -------------------------------------------------------------------------
    ax.set_title(title, fontsize=24, fontweight="bold", color=colors["text"], pad=20)
    ax.set_xlabel("Similarity", fontsize=22, color=colors["text"])
    ax.set_ylabel("Density", fontsize=22, color=colors["text"])
    ax.grid(True, alpha=0.3, color=colors["grid"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(colors["text"])
    ax.spines["bottom"].set_color(colors["text"])
    ax.tick_params(colors=colors["text"], which="both", labelsize=16)

    # -------------------------------------------------------------------------
    # Statistics box
    # -------------------------------------------------------------------------
    # stats_lines = [
    #     rf"{s['label']}: $\mu$={np.mean(s['values']):.4f}, $\sigma$={np.std(s['values']):.4f}"
    #     for s in series
    # ]
    # ax.text(
    #     0.02,
    #     0.98,
    #     "\n".join(stats_lines),
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     fontsize=12,
    #     bbox=dict(
    #         boxstyle="round,pad=0.5",
    #         facecolor="white",
    #         alpha=0.8,
    #         edgecolor=colors["grid"],
    #     ),
    # )

    # -------------------------------------------------------------------------
    # Legend (moved to figure level, bottom-center)
    # -------------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),  # x-centre, slightly below the axes
        ncol=len(labels),  # single row
        fontsize=16,
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, f"{tag}.pdf"),
        bbox_inches="tight",
        facecolor=colors["background"],
        format="pdf",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main loop -------------------------------------------------------------------
if __name__ == "__main__":
    for p in pairs:
        create_histogram(p["title"], p["tag"], p["series"])
