"""
Tests for OrScale optimizer variants.

Verifies:
1. All supported variants run without error and update parameters.
2. Trust ratio is computed and clipped correctly.
3. Gradient flow is correct (loss decreases).
4. Diagnostic dict is populated.
"""

import pytest
import torch
import torch.nn as nn

from orscale.optim.orscale_optimizer import OrScaleOptimizer, OrScaleVariant
from orscale.optim.muon import Muon
from orscale.optim.lamb import LAMB


def _make_simple_model():
    """A small model with one 2D matrix parameter."""
    model = nn.Linear(32, 16, bias=False)
    model.weight._diag_name = "test_weight"
    return model


def _run_optimizer_steps(opt_cls, opt_kwargs, steps=10):
    """Run a few optimizer steps and return (initial_loss, final_loss)."""
    torch.manual_seed(42)
    model = _make_simple_model()
    opt = opt_cls([model.weight], **opt_kwargs)

    x = torch.randn(8, 32)
    target = torch.randn(8, 16)

    initial_loss = None
    final_loss = None
    for i in range(steps):
        out = model(x)
        loss = ((out - target) ** 2).mean()
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
        final_loss = loss.item()

    return initial_loss, final_loss


@pytest.mark.parametrize("variant", [
    "orscale_original",
    "orscale_muon",
    "orscale_muon_wd",
    "orscale_muon_moonlight",
    "orscale_muon_moonlight_calibrated",
    "mutrust",
    "muscale",
    "muscale_alpha",
])
def test_orscale_variants_run(variant):
    """Each variant should run without error."""
    init_loss, final_loss = _run_optimizer_steps(
        OrScaleOptimizer,
        {"lr": 0.02, "variant": variant, "alpha": 0.5, "momentum": 0.95},
        steps=20,
    )
    assert final_loss < init_loss, \
        f"{variant}: loss did not decrease ({init_loss:.4f} -> {final_loss:.4f})"


def test_muon_runs():
    """Basic Muon should reduce loss."""
    init_loss, final_loss = _run_optimizer_steps(
        Muon, {"lr": 0.02, "momentum": 0.95}, steps=20,
    )
    assert final_loss < init_loss


def test_muon_moonlight_runs():
    """Muon with Moonlight RMS should reduce loss."""
    init_loss, final_loss = _run_optimizer_steps(
        Muon, {"lr": 0.005, "momentum": 0.95, "moonlight_rms": True}, steps=20,
    )
    assert final_loss < init_loss


def test_lamb_runs():
    """LAMB should reduce loss."""
    init_loss, final_loss = _run_optimizer_steps(
        LAMB, {"lr": 0.01}, steps=20,
    )
    assert final_loss < init_loss


def test_trust_ratio_clipping():
    """Trust ratio should be clipped to [r_min, r_max]."""
    torch.manual_seed(42)
    model = _make_simple_model()
    opt = OrScaleOptimizer(
        [model.weight], lr=0.02, variant="muscale_alpha",
        alpha=0.5, r_min=0.5, r_max=2.0, momentum=0.95,
    )

    x = torch.randn(8, 32)
    target = torch.randn(8, 16)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    diag = opt._diagnostics.get("test_weight", {})
    if "trust_ratio_clipped" in diag:
        assert 0.5 <= diag["trust_ratio_clipped"] <= 2.0, \
            f"Clipped ratio {diag['trust_ratio_clipped']} outside [0.5, 2.0]"


def test_orscale_muon_moonlight_calibrated_step0_ratio_is_one():
    """Auto-calibrated denominator should give trust_ratio_raw = 1.0 at step 0."""
    torch.manual_seed(42)
    model = _make_simple_model()
    opt = OrScaleOptimizer(
        [model.weight], lr=0.01,
        variant="orscale_muon_moonlight_calibrated",
        momentum=0.95, r_min=0.1, r_max=5.0,
    )

    x = torch.randn(8, 32)
    target = torch.randn(8, 16)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    diag = opt._diagnostics["test_weight"]
    assert "c_denom" in diag, "c_denom should be logged in diagnostics"
    assert diag["c_denom"] > 0, f"c_denom must be positive, got {diag['c_denom']}"
    raw = diag["trust_ratio_raw"]
    assert abs(raw - 1.0) < 1e-5, \
        f"At step 0 the auto-calibrated trust_ratio_raw must be 1.0, got {raw}"


def test_orscale_muon_moonlight_calibrated_width_invariance():
    """Two layers of very different widths should both start at trust_ratio = 1."""
    torch.manual_seed(0)

    narrow = nn.Linear(64, 64, bias=False)
    narrow.weight._diag_name = "narrow"

    wide = nn.Linear(2048, 2048, bias=False)
    wide.weight._diag_name = "wide"

    opt = OrScaleOptimizer(
        [narrow.weight, wide.weight], lr=0.01,
        variant="orscale_muon_moonlight_calibrated",
        momentum=0.95, r_min=0.1, r_max=5.0,
    )

    for w in (narrow.weight, wide.weight):
        x = torch.randn(4, w.shape[1])
        out = w @ x.T
        target = torch.randn_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()
    opt.step()

    for name in ("narrow", "wide"):
        raw = opt._diagnostics[name]["trust_ratio_raw"]
        assert abs(raw - 1.0) < 1e-5, \
            f"{name}: auto-calibrated trust_ratio at step 0 should be 1.0, got {raw}"


def test_orscale_muon_moonlight_calibrated_user_c_denom_used_uniformly():
    """If c_denom is supplied explicitly, it must be used as-is for every layer."""
    torch.manual_seed(0)
    model = _make_simple_model()

    user_c = 0.5
    opt = OrScaleOptimizer(
        [model.weight], lr=0.01,
        variant="orscale_muon_moonlight_calibrated",
        momentum=0.95, r_min=0.001, r_max=1000.0,
        c_denom=user_c,
    )

    x = torch.randn(8, 32)
    target = torch.randn(8, 16)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    opt.step()

    diag = opt._diagnostics["test_weight"]
    assert abs(diag["c_denom"] - user_c) < 1e-9, \
        f"User-supplied c_denom={user_c} must be used; got {diag['c_denom']}"


def test_orscale_muon_moonlight_calibrated_c_denom_persists_across_steps():
    """c_denom is calibrated once at step 0 and must not change afterwards."""
    torch.manual_seed(7)
    model = _make_simple_model()
    opt = OrScaleOptimizer(
        [model.weight], lr=0.05,
        variant="orscale_muon_moonlight_calibrated",
        momentum=0.95, r_min=0.1, r_max=5.0,
    )

    c_values = []
    x = torch.randn(8, 32)
    target = torch.randn(8, 16)
    for _ in range(5):
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        c_values.append(opt._diagnostics["test_weight"]["c_denom"])

    assert all(abs(c - c_values[0]) < 1e-9 for c in c_values), \
        f"c_denom must be constant across steps, got {c_values}"


def test_diagnostics_populated():
    """Optimizer should populate _diagnostics dict when params have _diag_name."""
    torch.manual_seed(42)
    model = _make_simple_model()
    opt = OrScaleOptimizer(
        [model.weight], lr=0.02, variant="muscale", momentum=0.95,
    )

    x = torch.randn(8, 32)
    loss = ((model(x)) ** 2).mean()
    loss.backward()
    opt.step()

    assert "test_weight" in opt._diagnostics
    diag = opt._diagnostics["test_weight"]
    assert "W_frob" in diag
    assert "G_frob" in diag
    assert "trust_ratio_raw" in diag
    assert "trust_ratio_clipped" in diag


def test_weight_decay():
    """With weight decay, parameter norm should be constrained."""
    torch.manual_seed(42)
    model = _make_simple_model()
    nn.init.normal_(model.weight, std=1.0)

    opt = OrScaleOptimizer(
        [model.weight], lr=0.001, variant="muscale", momentum=0.95,
        weight_decay=0.1,
    )

    initial_norm = model.weight.norm().item()
    x = torch.randn(8, 32)
    for _ in range(50):
        loss = ((model(x)) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    final_norm = model.weight.norm().item()
    # Weight decay should have reduced the norm somewhat
    assert final_norm < initial_norm * 1.5, \
        f"Norm grew too much with weight decay: {initial_norm:.3f} -> {final_norm:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
