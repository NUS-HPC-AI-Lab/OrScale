"""
Vision models for OrScale experiments.

Provides:
    - DavidNet: 9-layer residual ConvNet used in DAWNBench CIFAR-10 and the
        LAMB CIFAR table (Tables 6/7 of arXiv:1904.00962). Also used as the
        Muon-post CIFAR-10 speedrun reference model.
    - ResNet50: thin wrapper over ``torchvision.models.resnet50`` used for
        LAMB ImageNet/ResNet-50 large-batch experiments (Table 3 and 5).
    - PreActResNet20: small CIFAR baseline (He et al. 2016).

All models tag their trainable weight tensors with a ``muon_class`` attribute
(``"matrix"`` or ``"nonmatrix"``) so that ``build_optimizer`` routes them to
the correct optimizer. Following Keller Jordan's blog, the first (stem) conv
and the final classifier Linear are kept on AdamW, while all hidden Conv2d
and Linear weights are routed to the Muon family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Parameter labeling helpers
# ---------------------------------------------------------------------------

def _label_param(p: nn.Parameter, cls: str, name: str | None = None) -> None:
    p.muon_class = cls
    if name is not None:
        p._diag_name = name


def _label_hidden_conv_or_linear(module: nn.Module, prefix: str = "") -> None:
    """Tag all Conv2d / Linear weights as ``matrix`` and biases / norm params as ``nonmatrix``.

    Caller is responsible for re-tagging the stem conv and head Linear as
    ``nonmatrix`` afterwards (they should use AdamW, per Keller's blog).
    """
    for name, m in module.named_modules():
        full_name = f"{prefix}.{name}" if prefix and name else (prefix or name)
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            _label_param(m.weight, "matrix", name=f"{full_name}.weight" if full_name else "weight")
            if getattr(m, "bias", None) is not None:
                _label_param(m.bias, "nonmatrix", name=f"{full_name}.bias" if full_name else "bias")
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                _label_param(m.weight, "nonmatrix")
            if m.bias is not None:
                _label_param(m.bias, "nonmatrix")


# ---------------------------------------------------------------------------
# DavidNet (DAWNBench CIFAR-10 speedrun model)
# ---------------------------------------------------------------------------

class _ConvBNRelu(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _ResidualBlock(nn.Module):
    """Two ConvBNReLU layers with a residual connection (like DavidNet)."""

    def __init__(self, c: int):
        super().__init__()
        self.block = nn.Sequential(_ConvBNRelu(c, c), _ConvBNRelu(c, c))

    def forward(self, x):
        return x + self.block(x)


class DavidNet(nn.Module):
    """
    9-layer residual ConvNet for CIFAR-10 (DAWNBench speedrun model).

    Reference: https://github.com/davidcpage/cifar10-fast (DavidNet)
    Architecture (approximate):
        Prep:   ConvBNReLU(3, 64)
        Layer1: ConvBNReLU(64, 128) + MaxPool(2) + ResidualBlock(128)
        Layer2: ConvBNReLU(128, 256) + MaxPool(2)
        Layer3: ConvBNReLU(256, 512) + MaxPool(2) + ResidualBlock(512)
        Head:   AdaptiveMaxPool(1) + Linear(512, num_classes) * 0.125
    """

    def __init__(self, num_classes: int = 10, scale_head: float = 0.125):
        super().__init__()
        self.prep = _ConvBNRelu(3, 64)
        self.layer1 = nn.Sequential(
            _ConvBNRelu(64, 128),
            nn.MaxPool2d(2),
            _ResidualBlock(128),
        )
        self.layer2 = nn.Sequential(
            _ConvBNRelu(128, 256),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            _ConvBNRelu(256, 512),
            nn.MaxPool2d(2),
            _ResidualBlock(512),
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(512, num_classes, bias=False)
        self.scale_head = scale_head

        self._label_parameters()

    def _label_parameters(self) -> None:
        _label_hidden_conv_or_linear(self)
        # Stem conv and final head go to AdamW (Keller's rule).
        self.prep.conv.weight.muon_class = "nonmatrix"
        self.head.weight.muon_class = "nonmatrix"
        self.head.weight._diag_name = "head.weight"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.head(x) * self.scale_head

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# PreAct ResNet-20 (small CIFAR baseline)
# ---------------------------------------------------------------------------

class _PreActBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut: nn.Module
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + shortcut


class PreActResNet20(nn.Module):
    """He et al. (2016) PreActResNet-20 for CIFAR-10/100."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.head = nn.Linear(64, num_classes)
        self._label_parameters()

    @staticmethod
    def _make_layer(c_in, c_out, n_blocks, stride):
        layers = [_PreActBlock(c_in, c_out, stride)]
        for _ in range(n_blocks - 1):
            layers.append(_PreActBlock(c_out, c_out, 1))
        return nn.Sequential(*layers)

    def _label_parameters(self) -> None:
        _label_hidden_conv_or_linear(self)
        self.stem.weight.muon_class = "nonmatrix"
        self.head.weight.muon_class = "nonmatrix"
        self.head.weight._diag_name = "head.weight"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# ResNet-50 (torchvision wrapper)
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int = 1000, zero_init_residual: bool = True) -> nn.Module:
    """
    Build a ResNet-50 using torchvision and tag params for the Muon family.

    Args:
        num_classes: Number of classification classes (1000 for ImageNet-1K).
        zero_init_residual: If True, zero-initialize the last BN of each
            residual branch (He et al. 2016, Goyal et al. 2017 recipe).

    Returns:
        A ``torchvision.models.resnet50`` instance with ``muon_class`` labels.
    """
    try:
        from torchvision.models import resnet50
    except ImportError as err:
        raise ImportError(
            "torchvision is required for ResNet-50. Install with `pip install torchvision`."
        ) from err

    model = resnet50(weights=None, num_classes=num_classes)

    if zero_init_residual:
        # Goyal et al. (2017) recipe: zero-init the final BN in each Bottleneck
        # so the residual branch starts as identity.
        for m in model.modules():
            if hasattr(m, "bn3") and isinstance(m.bn3, nn.BatchNorm2d):
                nn.init.zeros_(m.bn3.weight)

    _label_hidden_conv_or_linear(model)
    # Stem conv (conv1) and final classifier (fc) -> AdamW (Keller's rule)
    model.conv1.weight.muon_class = "nonmatrix"
    model.fc.weight.muon_class = "nonmatrix"
    model.fc.weight._diag_name = "fc.weight"
    if model.fc.bias is not None:
        model.fc.bias.muon_class = "nonmatrix"

    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@dataclass
class VisionModelConfig:
    name: str = "davidnet"
    num_classes: int = 10


def build_vision_model(config: dict) -> nn.Module:
    """Factory that builds a vision model by name.

    Supported:
        - ``davidnet`` (CIFAR-10 speedrun)
        - ``preact_resnet20`` (CIFAR baseline)
        - ``resnet50`` (ImageNet)
    """
    name = config.get("name", "davidnet").lower()
    num_classes = int(config.get("num_classes", 10))

    if name == "davidnet":
        return DavidNet(num_classes=num_classes, scale_head=config.get("scale_head", 0.125))
    if name == "preact_resnet20":
        return PreActResNet20(num_classes=num_classes)
    if name == "resnet50":
        return build_resnet50(
            num_classes=num_classes,
            zero_init_residual=config.get("zero_init_residual", True),
        )
    raise ValueError(
        f"Unknown vision model: {name}. Choose from davidnet, preact_resnet20, resnet50."
    )
