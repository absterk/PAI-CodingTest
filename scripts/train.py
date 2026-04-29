"""Training entry point.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --override epochs=1 wandb_mode=disabled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root / src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pai.config import load_config, parse_overrides  # noqa: E402
from pai.trainer import Trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="PAI-CodingTest trainer")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config keys, e.g. epochs=1 loss_type=l1_ssim",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
