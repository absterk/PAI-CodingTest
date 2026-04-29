"""Inference on a split (val or test). Saves per-case metrics CSV.

Usage:
    python scripts/infer.py \
        --checkpoint checkpoints/<run>/checkpoint_best_ssim.pth \
        --split val \
        --config configs/baseline.yaml \
        --output-dir ./inference_results

The --config must match what was used for training (arch/backbone/etc.),
or be overridden to match. The config used for training is also archived
as `config.json` inside the checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pai.config import load_config, parse_overrides, TrainConfig  # noqa: E402
from pai.inference import run_inference, summarize, format_summary  # noqa: E402
from pai.visualize import plot_percentile_cases  # noqa: E402


def _load_cfg_for_checkpoint(checkpoint_path: Path, fallback_config: str | None, overrides: dict) -> TrainConfig:
    """Prefer config.json archived with checkpoint; fall back to --config."""
    archived = checkpoint_path.parent / "config.json"
    if archived.exists():
        with open(archived) as f:
            archived_cfg = json.load(f)
        archived_cfg.update(overrides)
        return TrainConfig(**archived_cfg)
    if fallback_config is None:
        raise FileNotFoundError(
            f"Neither {archived} exists nor --config was provided."
        )
    return load_config(fallback_config, overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAI-CodingTest inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default=None,
                        help="Fallback config if checkpoint's config.json missing.")
    parser.add_argument("--output-dir", type=str, default="./inference_results")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip percentile visualization figure.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    overrides = parse_overrides(args.override)
    cfg = _load_cfg_for_checkpoint(ckpt, args.config, overrides)

    run_id = ckpt.parent.name
    out_dir = Path(args.output_dir) / run_id / args.split
    results, cache = run_inference(cfg, args.split, ckpt, out_dir)

    stats = summarize(results)
    summary = format_summary(stats, args.split, len(results))
    print("\n" + summary + "\n")
    with open(out_dir / f"{args.split}_summary.txt", "w") as f:
        f.write(summary + "\n")
    with open(out_dir / f"{args.split}_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    if not args.no_viz:
        viz_path = out_dir / f"{args.split}_percentile_visualization.png"
        plot_percentile_cases(cache, viz_path, title_suffix=f"{args.split} set")
        print(f"Visualization saved to: {viz_path}")


if __name__ == "__main__":
    main()
