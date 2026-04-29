"""Loss factory + registry. All six losses from the original codebase are
preserved and selectable via TrainConfig.loss_type.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .config import TrainConfig
from .metrics import get_mask, ssim_map

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossBuilder = Callable[[TrainConfig, str], LossFn]

LOSS_FACTORY: Dict[str, LossBuilder] = {}


def register_loss(name: str) -> Callable[[LossBuilder], LossBuilder]:
    def decorator(fn: LossBuilder) -> LossBuilder:
        LOSS_FACTORY[name] = fn
        return fn
    return decorator


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim_map(pred, target).mean()


@register_loss("l1")
def _build_l1(cfg: TrainConfig, device: str) -> LossFn:
    base = nn.L1Loss()

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return base(pred, target)

    return fn


@register_loss("roi_weighted")
def _build_roi_weighted(cfg: TrainConfig, device: str) -> LossFn:
    roi_w = cfg.roi_weight

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = get_mask(target)
        weight_map = 1.0 + (roi_w - 1.0) * mask
        return (torch.abs(pred - target) * weight_map).mean()

    return fn


@register_loss("l1_ssim")
def _build_l1_ssim(cfg: TrainConfig, device: str) -> LossFn:
    ssim_w = cfg.ssim_weight
    base = nn.L1Loss()

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1.0 - ssim_w) * base(pred, target) + ssim_w * _ssim_loss(pred, target)

    return fn


class _VGGPerceptualLoss(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        self.feats = nn.Sequential(*list(vgg.children())[:16]).to(device)
        self.feats.eval()
        for p in self.feats.parameters():
            p.requires_grad = False
        self.adapt = nn.Conv2d(2, 3, 1, bias=False).to(device)
        nn.init.kaiming_normal_(self.adapt.weight)
        for p in self.adapt.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.feats(self.adapt(pred)), self.feats(self.adapt(target)))


@register_loss("l1_ssim_vgg")
def _build_l1_ssim_vgg(cfg: TrainConfig, device: str) -> LossFn:
    ssim_w = cfg.ssim_weight
    vgg_w = cfg.vgg_weight
    base = nn.L1Loss()
    vgg = _VGGPerceptualLoss(device)

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = base(pred, target)
        ssim_l = _ssim_loss(pred, target)
        vgg_l = vgg(pred, target)
        return (1.0 - ssim_w - vgg_w) * l1 + ssim_w * ssim_l + vgg_w * vgg_l

    return fn


@register_loss("amp_l1")
def _build_amp_l1(cfg: TrainConfig, device: str) -> LossFn:
    alpha = cfg.amp_weight_alpha

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t_max = target.max() + 1e-8
        weight = 1.0 + alpha * (target.detach() / t_max)
        return (torch.abs(pred - target) * weight).mean()

    return fn


@register_loss("amp_l1_ssim")
def _build_amp_l1_ssim(cfg: TrainConfig, device: str) -> LossFn:
    alpha = cfg.amp_weight_alpha
    ssim_w = cfg.ssim_weight

    def fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t_max = target.max() + 1e-8
        weight = 1.0 + alpha * (target.detach() / t_max)
        amp = (torch.abs(pred - target) * weight).mean()
        return (1.0 - ssim_w) * amp + ssim_w * _ssim_loss(pred, target)

    return fn


def build_criterion(cfg: TrainConfig, device: str) -> LossFn:
    if cfg.loss_type not in LOSS_FACTORY:
        raise ValueError(
            f"Unknown loss_type: {cfg.loss_type!r}. Registered: {sorted(LOSS_FACTORY)}"
        )
    return LOSS_FACTORY[cfg.loss_type](cfg, device)
