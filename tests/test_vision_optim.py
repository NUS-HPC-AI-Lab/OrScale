"""
Tests for Muon-family support of 4D Conv2d weights.

Verifies:
1. _split_params routes Conv2d weights to the matrix group.
2. Muon / OrScale optimizer steps reduce CE loss on a small ConvNet.
3. The reshape round-trip preserves the original parameter shape.
4. Explicit ``muon_class`` overrides still work for first conv / last linear.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orscale.optim import _split_params, build_optimizer
from orscale.optim.muon import Muon
from orscale.optim.orscale_optimizer import OrScaleOptimizer


class TinyConvNet(nn.Module):
    """Tiny ConvNet for testing: 2 conv layers + 1 FC head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(4, 16)
        self.head = nn.Linear(16 * 8 * 8, num_classes, bias=True)

        # Label parameters for optimizer routing
        self.stem.weight.muon_class = "nonmatrix"       # first conv -> AdamW
        self.conv.weight.muon_class = "matrix"          # hidden conv -> Muon
        self.head.weight.muon_class = "nonmatrix"       # output layer -> AdamW
        self.head.weight._diag_name = "head.weight"
        self.conv.weight._diag_name = "conv.weight"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = F.relu(self.norm(self.conv(x)))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        return self.head(x.flatten(1))


def _make_batch(batch: int = 4):
    torch.manual_seed(0)
    x = torch.randn(batch, 3, 8, 8)
    y = torch.randint(0, 10, (batch,))
    return x, y


def test_split_params_routes_conv_to_matrix():
    """_split_params should send 4D Conv2d weights to the matrix group."""
    model = TinyConvNet()
    matrix_params, nonmatrix_params = _split_params(model)

    matrix_ids = {id(p) for p in matrix_params}
    nonmatrix_ids = {id(p) for p in nonmatrix_params}

    # conv.weight is explicitly matrix (4D)
    assert id(model.conv.weight) in matrix_ids
    # stem.weight is explicit nonmatrix (first conv -> AdamW)
    assert id(model.stem.weight) in nonmatrix_ids
    # head.weight is explicit nonmatrix
    assert id(model.head.weight) in nonmatrix_ids
    # head.bias is 1D -> nonmatrix
    assert id(model.head.bias) in nonmatrix_ids
    # GroupNorm weight/bias are 1D -> nonmatrix
    assert id(model.norm.weight) in nonmatrix_ids


def test_split_params_defaults_4d_conv_to_matrix():
    """Without explicit muon_class, 4D conv weights should default to matrix."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.Conv2d(8, 16, 3, padding=1, bias=False),
    )
    matrix_params, _ = _split_params(model)
    matrix_ids = {id(p) for p in matrix_params}
    for m in model:
        assert id(m.weight) in matrix_ids


def test_muon_reshape_roundtrip_preserves_shape():
    """A Muon step on a 4D conv weight must return a tensor of the original shape."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
    original_shape = conv.weight.shape
    assert original_shape == (16, 8, 3, 3)

    opt = Muon([conv.weight], lr=1e-3, momentum=0.9, ns_iters=5)
    x = torch.randn(2, 8, 4, 4)
    target = torch.randn(2, 16, 4, 4)

    out = conv(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    assert conv.weight.shape == original_shape
    assert not torch.isnan(conv.weight).any()


def test_muon_reduces_loss_on_convnet():
    """Muon on hidden conv + AdamW on stem/head should reduce CE loss."""
    torch.manual_seed(0)
    model = TinyConvNet()
    x, y = _make_batch()

    config = {
        "lr": 0.02,
        "momentum": 0.95,
        "weight_decay": 0.0,
        "adamw_lr": 1e-3,
        "ns_iters": 5,
    }
    opts = build_optimizer("muon", model, config)
    assert isinstance(opts, list) and len(opts) == 2

    losses = []
    for _ in range(20):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.parametrize("variant", ["muon_moonlight", "muscale", "muscale_alpha"])
def test_orscale_variants_reduce_loss_on_convnet(variant):
    """OrScale variants should also reduce loss on a conv model."""
    torch.manual_seed(0)
    model = TinyConvNet()
    x, y = _make_batch()

    config = {
        "lr": 0.02,
        "momentum": 0.95,
        "weight_decay": 0.0,
        "alpha": 0.5,
        "r_min": 0.1,
        "r_max": 10.0,
        "eps": 1e-6,
        "ns_iters": 5,
        "adamw_lr": 1e-3,
    }
    opts = build_optimizer(variant, model, config)
    assert isinstance(opts, list) and len(opts) == 2

    losses = []
    for _ in range(20):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"{variant} loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


def test_conv_weight_updates_produce_orthogonal_like_direction():
    """
    After one Muon step the flattened 2D update should have near-unit
    max singular value (since the Newton-Schulz output is approximately orthogonal).
    """
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
    w0 = conv.weight.detach().clone()

    opt = Muon([conv.weight], lr=0.01, momentum=0.9, ns_iters=5, moonlight_rms=False)

    x = torch.randn(4, 16, 4, 4)
    target = torch.randn(4, 16, 4, 4)
    loss = ((conv(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    delta = (conv.weight.detach() - w0).view(conv.weight.shape[0], -1).float()
    # Muon update magnitude: ||Q||_F ~= sqrt(min(m, n)) * lr (without moonlight)
    # The normalized update (delta / lr) should have spectral norm ~= 1.
    # Use a loose bound since bf16 NS approximates the polar factor.
    sigma = torch.linalg.svdvals(delta / 0.01)
    assert sigma.max().item() < 1.5
    assert sigma.max().item() > 0.5
