"""U-Net with a pretrained ResNet encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from . import register_model


_CHANNELS = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}
_WEIGHTS = {
    "resnet18": tv_models.ResNet18_Weights.DEFAULT,
    "resnet34": tv_models.ResNet34_Weights.DEFAULT,
    "resnet50": tv_models.ResNet50_Weights.DEFAULT,
}
_MODEL_FN = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
}


@register_model("resnet_unet")
class ResNetUNet(nn.Module):
    """ResNet encoder + U-Net decoder.

    First conv is adapted from 3-channel RGB to 2-channel (755nm, 808nm) input
    by copying the first two input-channel weights from the pretrained kernel.
    """

    def __init__(self, n_class: int = 2, backbone: str = "resnet18") -> None:
        super().__init__()
        if backbone not in _CHANNELS:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        c0, c1, c2, c3, c4 = _CHANNELS[backbone]

        base = _MODEL_FN[backbone](weights=_WEIGHTS[backbone])
        base_layers = list(base.children())

        self.layer0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.layer0.weight[:, :2] = base.conv1.weight[:, :2]
        self.layer0_bn = base_layers[1]
        self.layer0_relu = base_layers[2]
        self.maxpool = base_layers[3]
        self.layer1 = base_layers[4]
        self.layer2 = base_layers[5]
        self.layer3 = base_layers[6]
        self.layer4 = base_layers[7]

        self.up1 = self._conv_block(c4 + c3, c3)
        self.up2 = self._conv_block(c3 + c2, c2)
        self.up3 = self._conv_block(c2 + c1, c1)
        self.up4 = self._conv_block(c1 + c0, c0)
        self.out_conv = nn.Conv2d(c0, n_class, kernel_size=1)

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0_relu(self.layer0_bn(self.layer0(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.up1(torch.cat([y, x3], dim=1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.up2(torch.cat([y, x2], dim=1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.up3(torch.cat([y, x1], dim=1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.up4(torch.cat([y, x0], dim=1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(y)
