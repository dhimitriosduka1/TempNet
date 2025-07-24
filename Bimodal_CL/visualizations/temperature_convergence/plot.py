#!/usr/bin/env python3
import argparse
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.ticker import MaxNLocator
import numpy as np

POS_KEY = "train/positive_samples_avg_temperature"
NEG_KEY = "train/negative_samples_avg_temperature"
STEP_CANDIDATES = ["step", "_step", "global_step", "trainer/global_step"]


def fetch_history(run, page_size=1000):
    """Return full history as a DataFrame (no key filtering)."""
    return pd.DataFrame([row for row in run.scan_history(page_size=page_size)])


def annotate_last(ax, x, y, fmt="{:.2f}", dx_frac=0.01, dy_frac=0.0):
    """Annotate last (x,y) with text offset by axis-fraction."""
    if len(x) == 0 or len(y) == 0:
        return
    x_last = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
    y_last = y.iloc[-1] if hasattr(y, "iloc") else y[-1]
    x_rng = ax.get_xlim()
    y_rng = ax.get_ylim()
    dx = (x_rng[1] - x_rng[0]) * dx_frac
    dy = (y_rng[1] - y_rng[0]) * dy_frac
    ax.text(
        x_last + dx,
        y_last + dy,
        fmt.format(y_last),
        fontsize=7,
        ha="left",
        va="center",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        clip_on=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="dduka-max-planck-society")
    p.add_argument("--project", default="Bimodal_CL_CC3M")
    p.add_argument(
        "--runs",
        default="nf4e68gz,s45x190m",
        help="Comma-separated run IDs (short hash or full path).",
    )
    p.add_argument("--page_size", type=int, default=1000)
    p.add_argument("--smooth", type=int, default=1, help="Rolling window size.")
    p.add_argument("--save_fig", default=None, help="Path to save figure.")
    p.add_argument("--fmt", default="{:.2f}", help="Format for value annotations.")
    p.add_argument("--no_cache", action="store_true", help="Force fresh W&B download.")
    p.add_argument("--cache_dir", default="~/.cache/wandb_histories")

    # Tick / label controls
    p.add_argument("--x_ticks", type=int, default=5)
    p.add_argument("--y_ticks", type=int, default=4)
    p.add_argument("--tick_labelsize", type=int, default=14)
    p.add_argument("--tick_length", type=float, default=4.0)

    # Shared y-axis padding
    p.add_argument(
        "--y_pad_frac",
        type=float,
        default=0.02,
        help="Extra fractional padding added to common y-limits.",
    )

    args = p.parse_args()

    api = wandb.Api()
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]

    plt.style.use(["science", "no-latex"])
    # sharey ensures same start/end automatically, but we'll also set them manually
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    ax_pos, ax_neg = axes

    for ax, title in zip(
        axes, ["Positive Samples Avg. Temp.", "Negative Samples Avg. Temp."]
    ):
        ax.set_title(title, pad=6, fontsize=18)
        ax.set_xlabel("Step", fontsize=16)
        ax.grid(True, axis="y", alpha=0.3)
        ax.margins(x=0.02)

    ax_pos.set_ylabel("Avg. Per Batch Temp.", fontsize=16)

    # track global y-range
    y_vals = []

    for rid in run_ids:
        run_path = rid if "/" in rid else f"{args.entity}/{args.project}/{rid}"
        run = api.run(run_path)
        df = fetch_history(run, page_size=args.page_size)

        # Normalize step column
        step_col = next((c for c in STEP_CANDIDATES if c in df.columns), None)
        if step_col and step_col != "step":
            df = df.rename(columns={step_col: "step"})
        if "step" not in df.columns:
            df["step"] = range(len(df))

        # Smooth metrics if present
        for key in (POS_KEY, NEG_KEY):
            if key in df.columns and args.smooth > 1:
                df[key] = df[key].rolling(args.smooth, min_periods=1).mean()

        label = "Fixed LR" if rid == "nf4e68gz" else "Cosine Annealing LR"

        if POS_KEY in df.columns:
            m = df[POS_KEY].notna()
            x_pos = df.loc[m, "step"]
            y_pos = df.loc[m, POS_KEY]
            ax_pos.plot(x_pos, y_pos, label=label, linewidth=1.2)
            y_vals.append(y_pos.values)

        if NEG_KEY in df.columns:
            m = df[NEG_KEY].notna()
            x_neg = df.loc[m, "step"]
            y_neg = df.loc[m, NEG_KEY]
            ax_neg.plot(x_neg, y_neg, label=label, linewidth=1.2)
            y_vals.append(y_neg.values)

    # ---- Make both y-axes start/end at same values ----
    if y_vals:
        all_y = np.concatenate(y_vals)
        y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
        pad = (y_max - y_min) * args.y_pad_frac
        y_min -= pad
        y_max += pad
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # Fewer & bigger ticks
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=args.x_ticks, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=args.y_ticks))
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=args.tick_labelsize,
            length=args.tick_length,
        )

    # Shared legend at bottom
    handles, labels = ax_pos.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
        handletextpad=0.6,
        labelspacing=0.6,
        fontsize=16,
    )

    fig.subplots_adjust(bottom=0.35, wspace=0.35)
    fig.tight_layout()

    out_path = (
        args.save_fig
        if args.save_fig is not None
        else "/BS/dduka/work/projects/TempNet/Bimodal_CL/visualizations/temperature_convergence/temperature_convergence.pdf"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
