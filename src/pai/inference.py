"""Inference helpers: load a checkpoint, run on a split, produce metrics CSV."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import TrainConfig
from .data import build_dataloader
from .metrics import (
    full_mse,
    full_psnr,
    full_ssim,
    get_mask,
    masked_mae,
    masked_mse,
    masked_psnr,
    masked_ssim,
)
from .models import build_model
from .utils import get_logger, resolve_device


def load_model_from_checkpoint(checkpoint_path: str | Path, cfg: TrainConfig, device: str) -> torch.nn.Module:
    model = build_model(cfg.arch, n_class=cfg.n_class, backbone=cfg.backbone).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    cfg: TrainConfig,
    split: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Run inference on `split` and save per-case metrics CSV + predictions.

    Returns:
        (results, tensor_cache) where results is a list of metric dicts
        and tensor_cache contains cpu tensors for downstream visualization.
    """
    device = resolve_device(cfg.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pai.inference", output_dir / "inference_log.txt")
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Split: %s", split)
    logger.info("Device: %s", device)

    loader = build_dataloader(split, cfg)
    logger.info("Cases: %d", len(loader.dataset))

    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    results: List[Dict] = []
    cache: List[Dict] = []

    for i, (p, a, fname) in enumerate(loader):
        p, a = p.to(device), a.to(device)
        pred = model(p)
        mask = get_mask(a)

        row = {
            "filename": fname[0],
            "mae": masked_mae(pred, a, mask).item(),
            "mse": masked_mse(pred, a, mask).item(),
            "psnr": masked_psnr(pred, a, mask).item(),
            "ssim": masked_ssim(pred, a, mask).item(),
            "mse_full": full_mse(pred, a).item(),
            "psnr_full": full_psnr(pred, a).item(),
            "ssim_full": full_ssim(pred, a).item(),
        }
        results.append(row)
        cache.append({
            "filename": fname[0],
            "input": p.detach().cpu(),
            "gt": a.detach().cpu(),
            "pred": pred.detach().cpu(),
            **row,
        })
        if (i + 1) % 50 == 0:
            logger.info("  processed %d / %d", i + 1, len(loader.dataset))

    csv_path = output_dir / f"{split}_metrics.csv"
    fieldnames = ["filename", "mae", "mse", "psnr", "ssim", "mse_full", "psnr_full", "ssim_full"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %s", csv_path)

    return results, cache


def summarize(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std/median/Q25/Q75/min/max per metric."""
    arrays = {k: np.array([r[k] for r in results]) for k in results[0] if k != "filename"}
    stats = {}
    for k, arr in arrays.items():
        stats[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return stats


def format_summary(stats: Dict[str, Dict[str, float]], split: str, n: int) -> str:
    lines = [
        f"=== {split} split summary  (N={n}) ===",
        f"{'Metric':<10} {'Mean':>10} {'Std':>10} {'Median':>10} {'Q25':>10} {'Q75':>10} {'Min':>10} {'Max':>10}",
        "-" * 82,
    ]
    for metric, s in stats.items():
        lines.append(
            f"{metric:<10} {s['mean']:>10.5f} {s['std']:>10.5f} {s['median']:>10.5f} "
            f"{s['q25']:>10.5f} {s['q75']:>10.5f} {s['min']:>10.5f} {s['max']:>10.5f}"
        )
    return "\n".join(lines)
