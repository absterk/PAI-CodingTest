"""Generate reproducible train/val/test splits and copy .mat files into data/.

Usage:
    python scripts/make_splits.py \
        --source-root ..  \
        --output-root ./data \
        --seed 42

Split protocol (per task spec):
    - 2000 total examples, indexed 1..2000.
    - Random subset of 1000 -> 800 train + 200 val.
    - From the remaining 1000, random subset of 200 -> test.

Outputs:
    data/splits.json                - {"train": [...], "val": [...], "test": [...]}
    data/PressureMaps2wv/*.mat      - copies of 1200 input files
    data/GTabsMaps2wv/*.mat         - copies of 1200 target files
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_TOTAL = 2000
N_TRAINVAL_POOL = 1000
N_TRAIN = 800
N_VAL = 200
N_TEST = 200

INPUT_SUBDIR = "PressureMaps2wv"
TARGET_SUBDIR = "GTabsMaps2wv"
INPUT_PATTERN = "pressuremaps755nm808nm_{idx}.mat"
TARGET_PATTERN = "GTabsmaps755nm808nm_{idx}.mat"


def make_splits(seed: int) -> dict[str, List[int]]:
    """Return deterministic {train, val, test} index lists (1-based indices)."""
    rng = np.random.default_rng(seed)
    all_indices = np.arange(1, N_TOTAL + 1)
    rng.shuffle(all_indices)

    trainval_pool = all_indices[:N_TRAINVAL_POOL]
    holdout_pool = all_indices[N_TRAINVAL_POOL:]

    rng.shuffle(trainval_pool)
    train_idx = trainval_pool[:N_TRAIN]
    val_idx = trainval_pool[N_TRAIN : N_TRAIN + N_VAL]

    rng.shuffle(holdout_pool)
    test_idx = holdout_pool[:N_TEST]

    return {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def copy_files(
    indices: List[int],
    source_root: Path,
    output_root: Path,
) -> None:
    """Copy .mat file pairs for the given indices into output_root."""
    src_in_dir = source_root / INPUT_SUBDIR
    src_tg_dir = source_root / TARGET_SUBDIR
    dst_in_dir = output_root / INPUT_SUBDIR
    dst_tg_dir = output_root / TARGET_SUBDIR
    dst_in_dir.mkdir(parents=True, exist_ok=True)
    dst_tg_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        in_name = INPUT_PATTERN.format(idx=idx)
        tg_name = TARGET_PATTERN.format(idx=idx)
        src_in = src_in_dir / in_name
        src_tg = src_tg_dir / tg_name
        if not src_in.exists():
            raise FileNotFoundError(f"Missing input: {src_in}")
        if not src_tg.exists():
            raise FileNotFoundError(f"Missing target: {src_tg}")
        shutil.copy2(src_in, dst_in_dir / in_name)
        shutil.copy2(src_tg, dst_tg_dir / tg_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate splits and copy data.")
    parser.add_argument(
        "--source-root", type=Path, default=Path(".."),
        help="Root containing PressureMaps2wv/ and GTabsMaps2wv/.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("./data"),
        help="Destination root for copied data and splits.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-copy", action="store_true",
        help="Only write splits.json; do not copy .mat files.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    splits = make_splits(args.seed)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set) == N_TRAIN
    assert len(val_set) == N_VAL
    assert len(test_set) == N_TEST
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    splits_path = args.output_root / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Wrote splits -> %s", splits_path)
    logger.info("  train=%d  val=%d  test=%d", N_TRAIN, N_VAL, N_TEST)

    if args.skip_copy:
        logger.info("--skip-copy set; done.")
        return

    all_indices = splits["train"] + splits["val"] + splits["test"]
    logger.info("Copying %d .mat pairs from %s ...", len(all_indices), args.source_root)
    copy_files(all_indices, args.source_root, args.output_root)
    logger.info("Copy complete.")


if __name__ == "__main__":
    main()
