"""Evaluate a trained checkpoint on val + test splits.

Produces:
    - <output_dir>/<run_id>/{val,test}/<split>_metrics.csv
    - <output_dir>/<run_id>/{val,test}/<split>_summary.{txt,json}
    - <output_dir>/<run_id>/{val,test}/<split>_percentile_visualization.png
    - <output_dir>/<run_id>/split_metrics_boxplot.png  (per-case distributions)
    - <output_dir>/<run_id>/report.md     (quick markdown summary)

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/<run>/checkpoint_best_ssim.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pai.config import parse_overrides, TrainConfig, load_config  # noqa: E402
from pai.inference import run_inference, summarize, format_summary  # noqa: E402
from pai.visualize import plot_percentile_cases, plot_split_boxplots  # noqa: E402


def _load_cfg_for_checkpoint(checkpoint_path: Path, fallback_config: str | None, overrides: dict) -> TrainConfig:
    archived = checkpoint_path.parent / "config.json"
    if archived.exists():
        with open(archived) as f:
            archived_cfg = json.load(f)
        archived_cfg.update(overrides)
        return TrainConfig(**archived_cfg)
    if fallback_config is None:
        raise FileNotFoundError(f"No config found: {archived} missing and --config not given.")
    return load_config(fallback_config, overrides)


def _report_md(run_id: str, splits: dict[str, dict], boxplot_rel: str | None = None) -> str:
    lines = [f"# Evaluation Report: `{run_id}`", ""]
    if boxplot_rel is not None and len(splits) > 1:
        lines.append("## Per-case metric distributions")
        lines.append("")
        lines.append(f"![split boxplots]({boxplot_rel})")
        lines.append("")
    for split, stats in splits.items():
        lines.append(f"## {split} split")
        lines.append("")
        lines.append("| Metric | Mean | Std | Median | Q25 | Q75 | Min | Max |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for metric, s in stats.items():
            lines.append(
                f"| {metric} | {s['mean']:.5f} | {s['std']:.5f} | {s['median']:.5f} "
                f"| {s['q25']:.5f} | {s['q75']:.5f} | {s['min']:.5f} | {s['max']:.5f} |"
            )
        lines.append("")
        lines.append(f"![percentile viz]({split}/{split}_percentile_visualization.png)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on val + test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    overrides = parse_overrides(args.override)
    cfg = _load_cfg_for_checkpoint(ckpt, args.config, overrides)

    run_id = ckpt.parent.name
    root_out = Path(args.output_dir) / run_id
    all_stats: dict[str, dict] = {}
    split_metrics: dict[str, dict[str, list[float]]] = {}

    for split in args.splits:
        out_dir = root_out / split
        results, cache = run_inference(cfg, split, ckpt, out_dir)
        stats = summarize(results)
        all_stats[split] = stats
        split_metrics[split] = {
            metric: [float(r[metric]) for r in results] for metric in stats
        }

        summary = format_summary(stats, split, len(results))
        print("\n" + summary + "\n")
        with open(out_dir / f"{split}_summary.txt", "w") as f:
            f.write(summary + "\n")
        with open(out_dir / f"{split}_summary.json", "w") as f:
            json.dump(stats, f, indent=2)
        plot_percentile_cases(cache, out_dir / f"{split}_percentile_visualization.png",
                              title_suffix=f"{split} set")

    boxplot_rel: str | None = None
    if len(split_metrics) > 1:
        boxplot_path = root_out / "split_metrics_boxplot.png"
        plot_split_boxplots(
            split_metrics, boxplot_path,
            title=f"Per-case metric distributions: {' vs '.join(split_metrics.keys())}",
        )
        boxplot_rel = boxplot_path.name
        print(f"Boxplot: {boxplot_path}")

    report_path = root_out / "report.md"
    with open(report_path, "w") as f:
        f.write(_report_md(run_id, all_stats, boxplot_rel=boxplot_rel))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
