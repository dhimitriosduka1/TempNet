#!/usr/bin/env python3
"""
Side-by-side histograms of similarity distributions for
positive/negative image–text pairs at different training stages.

Changes in this version (2025-07-29)
------------------------------------
• **True histograms** (with optional KDE overlay) instead of assuming
  pre-computed densities.
• One PDF containing two sub-plots (positive | negative).
• New CLI flags:
  – `--bins`      : number of bins (default 30)
  – `--kde`       : add a smooth Gauss-KDE curve on top of each histogram
  – `--no_density`: plot raw counts instead of probability densities
• Output file name: `<output_dir>/similarity_distributions.pdf`
• All previous CLI flags (`--neg_subset`, `--seed`, ...) still work.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from scipy.stats import gaussian_kde

# -----------------------------------------------------------------------------
# Plotting style --------------------------------------------------------------
plt.style.use(["science"])
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.size": 9,  # base
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 9,
    }
)

COLORS = {
    "start": "#FF6B6B",
    "middle": "#1f77b4",
    "end": "#4ECDC4",
    "start_dark": "#E55555",
    "middle_dark": "#1c6699",
    "end_dark": "#45B7AA",
    "grid": "#E0E0E0",
    "text": "#2C3E50",
}

# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------


def load_pickle(path: Path):
    """Load a pickle file with `rb` mode."""
    with path.open("rb") as f:
        return pickle.load(f)


def maybe_sample(
    values: np.ndarray, subset: float | int | None, rng: np.random.Generator
):
    """Return `values` or a random subset thereof.

    Parameters
    ----------
    values : np.ndarray
        1-D array of similarity values.
    subset : float | int | None
        If `None`, return all values.
        If 0 < subset ≤ 1, treat as fraction of rows.
        Else treat as absolute number of rows.
    rng : np.random.Generator
        Random generator for reproducible subsampling.
    """
    if subset is None:
        return values
    n = len(values)
    k = (
        max(1, int(round(subset * n)))  # fraction
        if 0 < subset <= 1
        else int(subset)  # absolute count
    )
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return values[idx]


# -----------------------------------------------------------------------------
# Plotting --------------------------------------------------------------------


def create_histograms_side_by_side(
    pairs: list[dict[str, object]],
    out_path: Path,
    *,
    bins: int = 30,
    density: bool = True,
    add_kde: bool = False,
    alpha_fill: float = 0.6,
    alpha_edge: float = 0.9,
):
    """Draw two histograms (positive & negative) side-by-side and save.

    Parameters
    ----------
    pairs : list
        Output of :func:`build_pairs` (len == 2).
    out_path : pathlib.Path
        Where to save the resulting PDF.
    bins : int
        Number of histogram bins.
    density : bool
        If *True*, normalise bars so area = 1 (probability density).
    add_kde : bool
        If *True*, overlay a Gaussian-KDE curve.
    """
    if len(pairs) != 2:
        raise ValueError("Expected exactly two pair configs (positive, negative).")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    data_min, data_max = -0.5, 0.8

    for ax, cfg in zip(axes, pairs):
        for s in cfg["series"]:
            values = np.asarray(s["values"]).ravel()
            # Histogram bars
            ax.hist(
                values,
                bins=bins,
                range=(data_min, data_max),
                density=density,
                color=s["color"],
                alpha=alpha_fill,
                edgecolor=s["dark_color"],
                linewidth=1.2,
                label=s["label"],
            )
            # Optional KDE overlay
            if add_kde:
                kde = gaussian_kde(values)
                x = np.linspace(data_min, data_max, 400)
                ax.plot(
                    x, kde(x), color=s["dark_color"], linewidth=2.0, alpha=alpha_edge
                )

        # Axis styling
        ax.set_title(cfg["title"], pad=6, fontsize=18, color=COLORS["text"])
        ax.set_xlabel("Similarity", fontsize=16, color=COLORS["text"])
        ax.set_xlim(data_min, data_max)
        ax.grid(True, alpha=0.3, color=COLORS["grid"], linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Density" if density else "Count", fontsize=16, color=COLORS["text"]
    )

    # Shared legend (bottom-center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.11),
        fontsize=16,
        handletextpad=0.6,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Data preparation ------------------------------------------------------------


def build_pairs(
    datasets: dict[str, dict[str, np.ndarray]],
    neg_subset: float | int | None,
    rng: np.random.Generator,
):
    """Return configs for positive & negative similarity plots.

    The structure returned here is consumed by
    :func:`create_histograms_side_by_side`.
    """
    positive = {
        "title": "Positive Pairs",
        "series": [
            {
                "label": "Initial distribution",
                "values": datasets["sot"]["i2t_similarity_matrix_positive_pairs"]
                .cpu()
                .numpy(),
                "color": COLORS["start"],
                "dark_color": COLORS["start_dark"],
            },
            {
                "label": "Final (InfoNCE)",
                "values": datasets["eot_infonce"][
                    "i2t_similarity_matrix_positive_pairs"
                ]
                .cpu()
                .numpy(),
                "color": COLORS["middle"],
                "dark_color": COLORS["middle_dark"],
            },
            {
                "label": "Final (TeMo)",
                "values": datasets["eot_temo"]["i2t_similarity_matrix_positive_pairs"]
                .cpu()
                .numpy(),
                "color": COLORS["end"],
                "dark_color": COLORS["end_dark"],
            },
        ],
    }

    negative_series = [
        {
            "label": "Initial distribution",
            "values": datasets["sot"]["i2t_similarity_matrix_negative_pairs"]
            .cpu()
            .numpy(),
            "color": COLORS["start"],
            "dark_color": COLORS["start_dark"],
        },
        {
            "label": "Final (InfoNCE)",
            "values": datasets["eot_infonce"]["i2t_similarity_matrix_negative_pairs"]
            .cpu()
            .numpy(),
            "color": COLORS["middle"],
            "dark_color": COLORS["middle_dark"],
        },
        {
            "label": "Final (TeMo)",
            "values": datasets["eot_temo"]["i2t_similarity_matrix_negative_pairs"]
            .cpu()
            .numpy(),
            "color": COLORS["end"],
            "dark_color": COLORS["end_dark"],
        },
    ]

    if neg_subset is not None:
        for s in negative_series:
            s["values"] = maybe_sample(np.asarray(s["values"]).ravel(), neg_subset, rng)

    negative = {
        "title": "Negative Pairs",
        "series": negative_series,
    }

    return [positive, negative]


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Plot side-by-side similarity histograms.")
    p.add_argument(
        "--base_dir",
        default="/BS/dduka/work/projects/TempNet/Bimodal_CL/visualizations/temperature_assignments_values",
        help="Root directory containing temperature-assignment pickles.",
    )
    p.add_argument(
        "--output_dir",
        default="/BS/dduka/work/projects/TempNet/Bimodal_CL/visualizations/temperature_assignments_values/plots",
        help="Directory to save PDF plots.",
    )
    p.add_argument(
        "--neg_subset",
        type=float,
        default=None,
        help="Subset of *negative* pairs to plot (fraction if 0<x≤1, else absolute count).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    p.add_argument("--bins", type=int, default=100, help="Number of histogram bins.")
    p.add_argument(
        "--no_density",
        action="store_true",
        help="Plot raw counts instead of probability densities.",
    )
    p.add_argument(
        "--kde",
        action="store_true",
        help="Add a smooth Gaussian-KDE curve on top of each histogram.",
    )

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    base = Path(os.path.expanduser(args.base_dir))
    out_dir = Path(os.path.expanduser(args.output_dir))

    datasets = {
        "sot": load_pickle(base / "common/cc3m_temperature_assignments.pkl"),
        "eot_infonce": load_pickle(base / "infonce/cc3m_temperature_assignments.pkl"),
        "eot_temo": load_pickle(base / "temo/cc3m_temperature_assignments.pkl"),
    }

    pairs = build_pairs(datasets, args.neg_subset, rng)
    out_path = out_dir / "similarity_distributions.pdf"
    create_histograms_side_by_side(
        pairs,
        out_path,
        bins=args.bins,
        density=True,
        add_kde=args.kde,
    )


if __name__ == "__main__":
    main()
