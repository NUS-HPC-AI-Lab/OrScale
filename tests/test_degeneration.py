"""
Degeneration tests: verify that OrScale variants collapse to known baselines
under specific hyperparameter settings.

These are critical correctness tests from the research plan:
1. MuScale with r_min = r_max = 1.0 should match Muon + Moonlight RMS.
2. MuScale-alpha with alpha = 0.0 should match Muon + Moonlight RMS.
3. OrScale-original with EMA momentum matches the PDF formulation.
"""

import math

import pytest
import torch
import torch.nn as nn

from orscale.optim.muon import Muon
from orscale.optim.orscale_optimizer import OrScaleOptimizer


def _get_weight_after_steps(opt_cls, opt_kwargs, steps=5, seed=42):
    """Run optimizer for N steps on a fixed problem, return final weight."""
    torch.manual_seed(seed)
    w = nn.Parameter(torch.randn(64, 32))
    w._diag_name = "w"

    # Fix gradient sequence by using the same random data each step
    opt = opt_cls([w], **opt_kwargs)
    grads = [torch.randn(64, 32) for _ in range(steps)]

    for i in range(steps):
        w.grad = grads[i].clone()
        opt.step()
        opt.zero_grad()

    return w.data.clone()


def test_muscale_constant_ratio_matches_muon_moonlight():
    """
    MuScale with r_min = r_max = 1.0 forces the trust ratio to 1.0 at every step.
    The update becomes: W -= lr * 1.0 * sqrt(max(m,n)) * Q
    which should be identical to Muon with moonlight_rms=True.
    """
    lr = 0.02
    mu = 0.95
    wd = 0.0
    steps = 10

    w_muscale = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "muscale", "r_min": 1.0, "r_max": 1.0},
        steps=steps,
    )

    w_muon_ml = _get_weight_after_steps(
        Muon,
        {"lr": lr, "momentum": mu, "weight_decay": wd, "moonlight_rms": True},
        steps=steps,
    )

    diff = (w_muscale.float() - w_muon_ml.float()).norm()
    ref = w_muon_ml.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"MuScale(r=1) vs Muon+Moonlight relative diff = {rel_diff:.6f} (expected < 0.01)"


def test_muscale_alpha_zero_matches_muon_moonlight():
    """
    MuScale-alpha with alpha=0 makes the trust ratio = (...)^0 = 1.0 always.
    Should be identical to Muon + Moonlight RMS.
    """
    lr = 0.02
    mu = 0.95
    wd = 0.0
    steps = 10

    w_alpha0 = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "muscale_alpha", "alpha": 0.0,
         "r_min": 0.0, "r_max": 100.0},
        steps=steps,
    )

    w_muon_ml = _get_weight_after_steps(
        Muon,
        {"lr": lr, "momentum": mu, "weight_decay": wd, "moonlight_rms": True},
        steps=steps,
    )

    diff = (w_alpha0.float() - w_muon_ml.float()).norm()
    ref = w_muon_ml.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"MuScale-alpha(a=0) vs Muon+Moonlight relative diff = {rel_diff:.6f} (expected < 0.01)"


def test_mutrust_vs_muscale_differ():
    """MuTrust and MuScale should produce different updates (different norm formulations)."""
    lr = 0.02
    mu = 0.95
    steps = 10

    w_mutrust = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "mutrust"},
        steps=steps,
    )

    w_muscale = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "muscale"},
        steps=steps,
    )

    diff = (w_mutrust.float() - w_muscale.float()).norm()
    assert diff > 1e-3, \
        f"MuTrust and MuScale should differ but diff = {diff:.6f}"


def test_orscale_original_uses_ema():
    """
    OrScale-original uses EMA momentum M = beta*M + (1-beta)*G, not accumulating.
    After one step with M_0=0, M_1 = (1-beta)*G_1, whereas accumulating gives M_1 = G_1.
    So the first-step updates should differ between orscale_original and mutrust.
    """
    lr = 0.02
    mu = 0.95

    w_original = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "orscale_original"},
        steps=3,
    )

    w_mutrust = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "mutrust"},
        steps=3,
    )

    diff = (w_original.float() - w_mutrust.float()).norm()
    assert diff > 1e-3, \
        f"orscale_original and mutrust should use different momentum but diff = {diff:.6f}"


def test_alpha_interpolation():
    """
    MuScale-alpha with alpha=1.0 should equal MuScale.
    """
    lr = 0.02
    mu = 0.95
    steps = 10

    w_muscale = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "muscale"},
        steps=steps,
    )

    w_alpha1 = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "variant": "muscale_alpha", "alpha": 1.0},
        steps=steps,
    )

    diff = (w_muscale.float() - w_alpha1.float()).norm()
    ref = w_muscale.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"MuScale vs MuScale-alpha(a=1) relative diff = {rel_diff:.6f} (expected < 0.01)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
