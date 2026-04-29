"""Model factory + registry.

Add new architectures by decorating the class with @register_model('name'),
then build via build_model(arch, n_class, backbone).
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

MODEL_FACTORY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_FACTORY:
            raise KeyError(f"Model '{name}' already registered.")
        MODEL_FACTORY[name] = cls
        return cls
    return decorator


def build_model(arch: str, n_class: int = 2, backbone: str = "resnet18") -> nn.Module:
    if arch not in MODEL_FACTORY:
        raise ValueError(
            f"Unknown arch: {arch!r}. Registered: {sorted(MODEL_FACTORY)}"
        )
    cls = MODEL_FACTORY[arch]
    return cls(n_class=n_class, backbone=backbone)


# Import to trigger registrations
from . import resnet_unet  # noqa: F401,E402
from . import attention_unet  # noqa: F401,E402

__all__ = ["build_model", "register_model", "MODEL_FACTORY"]
