"""Qualitative evaluation: percentile-based visualization.

Produces a multi-row figure showing reconstruction cases at percentiles
[0, 25, 50, 75, 100] of the masked MAE distribution (best -> worst), and
box+strip plots comparing per-case metric distributions across splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable


def use_times_new_roman() -> None:
    """Set Times New Roman as the default matplotlib font family.

    Safe to call multiple times. Falls back to a serif face automatically
    if Times New Roman is not installed on the host.
    """
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"


use_times_new_roman()

METRIC_LAYOUT: Sequence[tuple[str, str]] = (
    ("mae", "MAE (masked)"),
    ("mse", "MSE (masked)"),
    ("psnr", "PSNR / dB (masked)"),
    ("ssim", "SSIM (masked)"),
    ("mse_full", "MSE (full image)"),
    ("psnr_full", "PSNR / dB (full image)"),
    ("ssim_full", "SSIM (full image)"),
)

SPLIT_COLORS: Mapping[str, str] = {
    "val": "#1f77b4",
    "test": "#d62728",
}

PERCENTILES = {
    "P0 (best)": 0.0,
    "P25": 0.25,
    "P50 (median)": 0.50,
    "P75": 0.75,
    "P100 (worst)": 1.0,
}


def _pick_percentile_cases(cache: List[Dict], key: str = "mae") -> List[Dict]:
    sorted_cases = sorted(cache, key=lambda x: x[key])
    n = len(sorted_cases)
    picks: List[Dict] = []
    for _, q in PERCENTILES.items():
        idx = int(round(q * (n - 1)))
        picks.append(sorted_cases[idx])
    return picks


def plot_percentile_cases(
    cache: List[Dict],
    output_path: str | Path,
    title_suffix: str = "",
    sort_metric: str = "mae",
) -> Path:
    """Render a 5-row x 8-col grid (Input/GT/Pred/Diff for 755nm + 808nm)."""
    picks = _pick_percentile_cases(cache, key=sort_metric)
    col_titles = [
        "Input 755nm", "GT 755nm", "Pred 755nm", "Diff 755nm",
        "Input 808nm", "GT 808nm", "Pred 808nm", "Diff 808nm",
    ]

    fig, axes = plt.subplots(5, 8, figsize=(36, 22))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.02, wspace=0.3, hspace=0.15)

    labels = list(PERCENTILES.keys())
    for row_idx, (label, case) in enumerate(zip(labels, picks)):
        inp = case["input"][0].numpy()
        gt = case["gt"][0].numpy()
        pred = case["pred"][0].numpy()
        diff_755 = np.abs(gt[0] - pred[0])
        diff_808 = np.abs(gt[1] - pred[1])
        images = [inp[0], gt[0], pred[0], diff_755, inp[1], gt[1], pred[1], diff_808]

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap="hot")
            ax.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            cax.tick_params(labelsize=9)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=14, fontweight="bold")

        pos = axes[row_idx, 0].get_position()
        y = pos.y0 + pos.height / 2
        text = (
            f"{label}\n{case['filename']}\n"
            f"MAE={case['mae']:.4f}  MSE={case['mse']:.5f}\n"
            f"PSNR={case['psnr']:.2f}  SSIM={case['ssim']:.4f}"
        )
        fig.text(
            0.01, y, text, fontsize=10, fontweight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.85),
        )

    suffix = f"  |  {title_suffix}" if title_suffix else ""
    plt.suptitle(
        f"Reconstruction at MAE percentiles [0, 25, 50, 75, 100]{suffix}",
        fontsize=18, fontweight="bold", y=0.98,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_split_boxplots(
    split_metrics: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: str | Path,
    title: str = "Per-case metric distributions",
    jitter: float = 0.08,
    point_alpha: float = 0.35,
    seed: int = 0,
) -> Path:
    """Draw box + jittered scatter plots comparing splits per metric.

    Args:
        split_metrics: {split_name: {metric_key: [values, ...]}}.
        output_path: where to save the PNG.
        title: figure suptitle.
        jitter: horizontal jitter half-width for the scatter points.
        point_alpha: scatter alpha.
        seed: RNG seed for reproducible jitter.
    """
    splits = list(split_metrics.keys())
    metrics = [m for m in METRIC_LAYOUT if all(m[0] in split_metrics[s] for s in splits)]
    if not metrics:
        raise ValueError("No shared metrics across splits to plot.")

    n_cols = 4
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(seed)

    for flat_idx, (key, label) in enumerate(metrics):
        ax = axes[flat_idx // n_cols, flat_idx % n_cols]
        data = [np.asarray(split_metrics[s][key], dtype=float) for s in splits]
        positions = np.arange(1, len(splits) + 1)

        bp = ax.boxplot(
            data, positions=positions, widths=0.55, showfliers=False,
            patch_artist=True, medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, s in zip(bp["boxes"], splits):
            color = SPLIT_COLORS.get(s, "#7f7f7f")
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps"):
            for line, s in zip(bp[element], np.repeat(splits, 2)):
                line.set_color(SPLIT_COLORS.get(s, "#7f7f7f"))

        for pos, values, s in zip(positions, data, splits):
            x = pos + rng.uniform(-jitter, jitter, size=values.shape)
            ax.scatter(
                x, values, s=10, alpha=point_alpha,
                color=SPLIT_COLORS.get(s, "#7f7f7f"), edgecolors="none",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(splits)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    for extra in range(len(metrics), n_rows * n_cols):
        axes[extra // n_cols, extra % n_cols].axis("off")

    legend_handles = [
        mpatches.Patch(color=SPLIT_COLORS.get(s, "#7f7f7f"), alpha=0.6, label=s)
        for s in splits
    ]
    fig.legend(
        handles=legend_handles, loc="upper right",
        bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=11,
    )
    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
