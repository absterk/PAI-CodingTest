"""Photoacoustic dataset and split-based dataloader construction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig

INPUT_SUBDIR = "PressureMaps2wv"
TARGET_SUBDIR = "GTabsMaps2wv"
INPUT_PATTERN = "pressuremaps755nm808nm_{idx}.mat"
TARGET_PATTERN = "GTabsmaps755nm808nm_{idx}.mat"


class PhotoacousticDataset(Dataset):
    """Dual-wavelength (755/808nm) photoacoustic input/target pairs.

    Each sample is loaded from .mat files, optionally standardized, target-scaled
    by A_MAX, zero-padded to img_size, and optionally augmented (flips + 90° rotations).
    """

    def __init__(
        self,
        input_files: List[str],
        target_files: List[str],
        cfg: TrainConfig,
        is_training: bool = False,
    ):
        assert len(input_files) == len(target_files)
        self.input_files = input_files
        self.target_files = target_files
        self.a_max = cfg.a_max
        self.target_size = cfg.img_size
        self.standardize = cfg.input_standardize
        self.augment = cfg.augment and is_training

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        in_path = self.input_files[idx]
        tg_path = self.target_files[idx]
        p_mat = scipy.io.loadmat(in_path)
        a_mat = scipy.io.loadmat(tg_path)

        p_key = [k for k in p_mat if not k.startswith("__")][0]
        a_key = [k for k in a_mat if not k.startswith("__")][0]
        p = p_mat[p_key]
        a = a_mat[a_key]

        if self.standardize:
            p = (p - p.mean()) / (p.std() + 1e-8)

        a = a / self.a_max

        h, w, _ = p.shape
        pad_h = (self.target_size - h) // 2
        pad_w = (self.target_size - w) // 2
        p = np.pad(p, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
        a = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

        if self.augment:
            if np.random.rand() > 0.5:
                p = np.flip(p, axis=1).copy()
                a = np.flip(a, axis=1).copy()
            if np.random.rand() > 0.5:
                p = np.flip(p, axis=0).copy()
                a = np.flip(a, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                p = np.rot90(p, k, axes=(0, 1)).copy()
                a = np.rot90(a, k, axes=(0, 1)).copy()

        p_t = torch.from_numpy(p.transpose(2, 0, 1)).float()
        a_t = torch.from_numpy(a.transpose(2, 0, 1)).float()
        return p_t, a_t, os.path.basename(in_path)


def load_splits(splits_path: str | Path) -> Dict[str, List[int]]:
    with open(splits_path) as f:
        return json.load(f)


def _paths_for_indices(data_root: Path, indices: List[int]) -> Tuple[List[str], List[str]]:
    in_dir = data_root / INPUT_SUBDIR
    tg_dir = data_root / TARGET_SUBDIR
    inputs = [str(in_dir / INPUT_PATTERN.format(idx=i)) for i in indices]
    targets = [str(tg_dir / TARGET_PATTERN.format(idx=i)) for i in indices]
    return inputs, targets


def build_dataset(split: str, cfg: TrainConfig) -> PhotoacousticDataset:
    """Build dataset for split in {'train','val','test'}."""
    splits = load_splits(cfg.splits_path)
    if split not in splits:
        raise KeyError(f"Unknown split: {split}. Known: {list(splits.keys())}")
    inputs, targets = _paths_for_indices(Path(cfg.data_root), splits[split])
    return PhotoacousticDataset(inputs, targets, cfg, is_training=(split == "train"))


def build_dataloader(split: str, cfg: TrainConfig) -> DataLoader:
    dataset = build_dataset(split, cfg)
    shuffle = split == "train"
    batch_size = cfg.batch_size if split == "train" else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
