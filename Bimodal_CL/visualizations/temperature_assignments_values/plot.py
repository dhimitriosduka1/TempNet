import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up beautiful plotting style
# plt.style.use(["science", "no-latex"])
plt.style.use(["science", "no-latex"])
sns.set_palette("husl")

# Custom color palette - modern and sophisticated
colors = {
    "start": "#FF6B6B",  # Coral red
    "end": "#4ECDC4",  # Teal
    "start_dark": "#E55555",  # Darker coral
    "end_dark": "#45B7AA",  # Darker teal
    "background": "#FFFFFF",  # Light gray background
    "grid": "#E0E0E0",  # Light grid
    "text": "#2C3E50",  # Dark blue-gray text
}

# File paths (replace with your actual paths)
start_file_path = "/BS/dduka/work/projects/TempNet/Bimodal_CL/temperature_assignments_values/start_cc3m_temperature_assignments.pkl"
end_file_path = "/BS/dduka/work/projects/TempNet/Bimodal_CL/temperature_assignments_values/end_cc3m_temperature_assignments.pkl"

# Load data
with open(start_file_path, "rb") as f:
    start_data = pickle.load(f)

with open(end_file_path, "rb") as f:
    end_data = pickle.load(f)

min_possible_temp = start_data["min_possible_temp"]
max_possible_temp = start_data["max_possible_temp"]

# Extract temperature assignments
i2t_positive_temp_assignments_start = start_data["i2t_positive_temp_assignments"]
t2i_positive_temp_assignments_start = start_data["t2i_positive_temp_assignments"]
i2t_negative_temp_assignments_start = start_data["i2t_negative_temp_assignments"]
t2i_negative_temp_assignments_start = start_data["t2i_negative_temp_assignments"]

i2t_positive_temp_assignments_end = end_data["i2t_positive_temp_assignments"]
t2i_positive_temp_assignments_end = end_data["t2i_positive_temp_assignments"]
i2t_negative_temp_assignments_end = end_data["i2t_negative_temp_assignments"]
t2i_negative_temp_assignments_end = end_data["t2i_negative_temp_assignments"]

# Configuration
n_bins = 5
alpha_fill = 0.7
alpha_line = 0.9
output_dir = "/BS/dduka/work/projects/TempNet/Bimodal_CL/temperature_assignments_values"

# Define pairs with better titles - only i2t positive and negative
pairs = [
    (
        "Temperature Assigned to Positive Pairs",
        "i2t_positive",
        i2t_positive_temp_assignments_start,
        i2t_positive_temp_assignments_end,
    ),
    (
        "Temperature Assigned to Negative Pairs",
        "i2t_negative",
        i2t_negative_temp_assignments_start,
        i2t_negative_temp_assignments_end,
    ),
]

# Create output directory
import os

os.makedirs(output_dir, exist_ok=True)


# Function to create beautiful individual plots
def create_beautiful_histogram(title, tag, start_vals, end_vals):
    # Set up the figure with custom styling
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(colors["background"])
    ax.set_facecolor("white")

    # Calculate statistics for better binning
    all_vals = np.concatenate([start_vals, end_vals])
    data_min, data_max = min_possible_temp, max_possible_temp
    bins = np.linspace(data_min, data_max, n_bins)

    # Create histograms with smooth curves
    for vals, color, dark_color, label, offset in [
        (start_vals, colors["start"], colors["start_dark"], "Start of training", 0),
        (end_vals, colors["end"], colors["end_dark"], "End of training", 0),
    ]:
        # Histogram
        counts, bin_edges = np.histogram(vals, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth the histogram with KDE
        kde = gaussian_kde(vals)
        x_smooth = np.linspace(data_min, data_max, 200)
        y_smooth = kde(x_smooth)

        # Fill area under curve
        ax.fill_between(
            x_smooth, y_smooth, alpha=alpha_fill, color=color, label=label, linewidth=0
        )

        # Add smooth line on top
        ax.plot(x_smooth, y_smooth, color=dark_color, linewidth=2.5, alpha=alpha_line)

    # Styling
    ax.set_title(title, fontsize=20, fontweight="bold", color=colors["text"], pad=20)
    ax.set_xlabel("Temperature Value", fontsize=16, color=colors["text"])
    ax.set_ylabel("Density", fontsize=16, color=colors["text"])

    # Grid and spines
    ax.grid(True, alpha=0.3, color=colors["grid"], linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(colors["text"])
    ax.spines["bottom"].set_color(colors["text"])

    # Tick styling with larger font
    ax.tick_params(colors=colors["text"], which="both", labelsize=12)

    # Legend with custom styling - no shadow
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        fontsize=14,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(colors["grid"])

    # Add statistics text box with higher precision
    start_mean, start_std = np.mean(start_vals), np.std(start_vals)
    end_mean, end_std = np.mean(end_vals), np.std(end_vals)

    stats_text = (
        f"Start: $\\mu$={start_mean:.4f}, $\\sigma$={start_std:.4f}\n"
        f"End: $\\mu$={end_mean:.4f}, $\\sigma$={end_std:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.8,
            edgecolor=colors["grid"],
        ),
    )

    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"{title}.pdf")
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        facecolor=colors["background"],
        edgecolor="none",
        format="pdf",
    )
    plt.close(fig)


# Create individual beautiful plots
for title, tag, start_vals, end_vals in pairs:
    create_beautiful_histogram(title, tag, start_vals, end_vals)
