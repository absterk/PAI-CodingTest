"""Attention U-Net (from-scratch encoder with attention-gated skips).

Reference: Oktay et al., "Attention U-Net" (2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        return x * self.psi(self.relu(g1 + x1))


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("attention_unet")
class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 2, n_class: int = 2, init_features: int = 64) -> None:
        super().__init__()
        f = init_features
        self.enc1 = _ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(f * 8, f * 8, f * 4)
        self.dec4 = _ConvBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(f * 4, f * 4, f * 2)
        self.dec3 = _ConvBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(f * 2, f * 2, f)
        self.dec2 = _ConvBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(f, f, f // 2)
        self.dec1 = _ConvBlock(f * 2, f)

        self.out_conv = nn.Conv2d(f, n_class, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
