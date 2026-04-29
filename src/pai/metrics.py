"""Masked and full-image reconstruction metrics.

Targets are in [0, 1] after A_MAX scaling; this is the assumed dynamic range
for PSNR. All masked metrics use an ROI defined as pixels where the target has
non-zero signal (any wavelength channel).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def get_mask(target: torch.Tensor) -> torch.Tensor:
    """Return (B,1,H,W) float mask: 1 where any target channel has signal."""
    return (torch.abs(target).sum(dim=1, keepdim=True) > 1e-6).float()


def _apply_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast mask across channels and return (denom_elements, sq_err_tensor)."""
    channel = target.shape[1]
    denom = mask.sum() * channel + 1e-8
    return denom, mask


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom, _ = _apply_mask(pred, target, mask)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom, _ = _apply_mask(pred, target, mask)
    return ((pred - target) ** 2 * mask).sum() / denom


def full_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = masked_mse(pred, target, mask)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 10 * torch.log10((data_range ** 2) / mse)


def full_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = full_mse(pred, target)
    if mse.item() == 0:
        return torch.tensor(100.0, device=pred.device)
    return 10 * torch.log10((data_range ** 2) / mse)


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    g = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return g / g.sum()


def _create_window(window_size: int, channel: int, device: torch.device | str) -> torch.Tensor:
    w1 = _gaussian(window_size, 1.5).unsqueeze(1)
    w2 = w1.mm(w1.t()).float().unsqueeze(0).unsqueeze(0)
    return w2.expand(channel, 1, window_size, window_size).contiguous().to(device)


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channel = img1.shape[1]
    window = _create_window(window_size, channel, img1.device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )


def masked_ssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s_map = ssim_map(pred, target)
    channel = pred.shape[1]
    return (s_map * mask).sum() / (mask.sum() * channel + 1e-8)


def full_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ssim_map(pred, target).mean()
