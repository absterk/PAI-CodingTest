"""Visualize random dataset examples (no model required).

Randomly samples N cases from a split and draws a grid where each row is
one case and the 4 columns are the two input channels (755 nm, 808 nm)
followed by the two target channels (755 nm, 808 nm).

Usage:
    python scripts/visualize_examples.py \
        --split train --num-samples 10 --seed 42 \
        --output-path ./data_examples.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pai.config import load_config, parse_overrides  # noqa: E402
from pai.data import build_dataset  # noqa: E402
from pai.visualize import use_times_new_roman  # noqa: E402

use_times_new_roman()

COL_TITLES = ["Input 755 nm", "Input 808 nm", "GT 755 nm", "GT 808 nm"]


def _pick_indices(n_total: int, n_pick: int, seed: int) -> List[int]:
    if n_pick > n_total:
        raise ValueError(f"Requested {n_pick} samples but split has only {n_total}.")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=n_pick, replace=False).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize N random dataset examples")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="./data_examples.png")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    # Force raw visualization: disable augment + standardize regardless of config.
    overrides.setdefault("augment", False)
    overrides.setdefault("input_standardize", False)
    cfg = load_config(args.config, overrides)

    dataset = build_dataset(args.split, cfg)
    picks = _pick_indices(len(dataset), args.num_samples, args.seed)

    n_rows = len(picks)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.02, wspace=0.3, hspace=0.25)

    for row_idx, idx in enumerate(picks):
        p_t, a_t, fname = dataset[idx]
        p = p_t.numpy()  # (2, H, W)
        a = a_t.numpy()  # (2, H, W), already scaled by A_MAX
        panels = [p[0], p[1], a[0], a[1]]

        for col_idx, img in enumerate(panels):
            ax = axes[row_idx, col_idx]
            cmap = "seismic" if col_idx < 2 else "hot"
            if col_idx < 2:
                vmax = float(np.abs(img).max()) or 1.0
                im = ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            cax.tick_params(labelsize=8)
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=13, fontweight="bold")

        axes[row_idx, 0].text(
            -0.15, 0.5, f"#{idx}\n{fname}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=9, va="center", ha="right",
        )

    plt.suptitle(
        f"{args.num_samples} random examples from '{args.split}' (seed={args.seed})",
        fontsize=15, fontweight="bold", y=0.99,
    )

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {out}")


if __name__ == "__main__":
    main()
