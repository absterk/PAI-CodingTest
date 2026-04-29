"""Frozen dataclass configuration with YAML loader and CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class TrainConfig:
    # Data
    data_root: str = "./data"
    splits_path: str = "./data/splits.json"
    a_max: float = 0.02
    img_size: int = 128
    input_standardize: bool = False
    augment: bool = False

    # Model
    arch: str = "resnet_unet"             # {"resnet_unet", "attention_unet"}
    backbone: str = "resnet18"            # {"resnet18", "resnet34", "resnet50"}
    n_class: int = 2

    # Loss: {"l1", "roi_weighted", "l1_ssim", "l1_ssim_vgg", "amp_l1", "amp_l1_ssim"}
    loss_type: str = "l1"
    ssim_weight: float = 0.5
    vgg_weight: float = 0.1
    roi_weight: float = 5.0
    amp_weight_alpha: float = 3.0

    # Optim
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    early_stop_patience: int = 50
    grad_clip_norm: float = 1.0
    num_workers: int = 0

    # Runtime
    seed: int = 42
    device: str = "auto"                # {"auto", "cpu", "cuda", "mps"}

    # Logging / checkpoints
    save_root: str = "./checkpoints"
    run_name: str = ""                  # empty => auto-generated
    wandb_project: str = "PAI-CodingTest"
    wandb_mode: str = "online"          # {"online", "offline", "disabled"}


def _parse_scalar(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in value or "e" in low:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _coerce(field_type: type, value: Any) -> Any:
    if value is None:
        return value
    if field_type is bool and isinstance(value, str):
        return value.lower() == "true"
    try:
        return field_type(value)
    except (TypeError, ValueError):
        return value


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> TrainConfig:
    """Load YAML config then apply dot-key overrides (e.g., {'epochs': 1})."""
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {path} must be a YAML mapping.")
        data.update(loaded)
    if overrides:
        data.update(overrides)

    field_types = {f.name: f.type for f in TrainConfig.__dataclass_fields__.values()}
    known = {}
    for k, v in data.items():
        if k not in field_types:
            raise KeyError(f"Unknown config key: {k}")
        known[k] = _coerce(field_types[k], v)
    return TrainConfig(**known)


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse ['key=value', ...] into a dict (values auto-cast)."""
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Override must be key=value: {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = _parse_scalar(v.strip())
    return out


def config_to_dict(cfg: TrainConfig) -> Dict[str, Any]:
    return asdict(cfg)
