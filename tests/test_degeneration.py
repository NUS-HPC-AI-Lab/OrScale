"""
Degeneration tests: verify that OrScale variants collapse to known baselines
under specific hyperparameter settings.

These are critical correctness tests from the research plan:
1. MuScale with r_min = r_max = 1.0 should match Muon + Moonlight RMS.
2. MuScale-alpha with alpha = 0.0 should match Muon + Moonlight RMS.
3. OrScale-original with EMA momentum matches the PDF formulation.
4. New trust-on-baseline variants should reduce to their Muon baselines when r = 1.
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


def test_orscale_muon_constant_ratio_matches_muon():
    """OrScale-Muon with r=1 should exactly recover standard Muon."""
    lr = 0.02
    mu = 0.95
    wd = 0.0
    steps = 10

    w_orscale_muon = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon", "r_min": 1.0, "r_max": 1.0},
        steps=steps,
    )

    w_muon = _get_weight_after_steps(
        Muon,
        {"lr": lr, "momentum": mu, "weight_decay": wd, "moonlight_rms": False},
        steps=steps,
    )

    diff = (w_orscale_muon.float() - w_muon.float()).norm()
    ref = w_muon.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"OrScale-Muon(r=1) vs Muon relative diff = {rel_diff:.6f} (expected < 0.01)"


def test_orscale_muon_moonlight_constant_ratio_matches_muon_moonlight():
    """
    OrScale-Muon-Moonlight and stock Muon+Moonlight both apply the full
    Moonlight RMS-matching scale 0.2 * sqrt(max(m, n)) to Q. With r=1 and wd=0,
    OrScale-Muon-Moonlight's coupled update reduces to:

        W -= lr * 0.2 * sqrt(max(m, n)) * Q

    which is identical to Muon + Moonlight (wd=0).
    """
    lr = 0.02
    mu = 0.95
    wd = 0.0
    steps = 10

    w_orscale_ml = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon_moonlight", "r_min": 1.0, "r_max": 1.0},
        steps=steps,
    )

    w_muon_ml = _get_weight_after_steps(
        Muon,
        {"lr": lr, "momentum": mu, "weight_decay": wd, "moonlight_rms": True},
        steps=steps,
    )

    diff = (w_orscale_ml.float() - w_muon_ml.float()).norm()
    ref = w_muon_ml.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"OrScale-Muon-Moonlight(r=1) vs Muon+Moonlight relative diff = {rel_diff:.6f} (expected < 0.01)"


def test_orscale_muon_moonlight_couples_weight_decay():
    """
    OrScale-Muon-Moonlight folds weight decay *inside* the trust-ratio scaling:

        W -= lr * r_hat * (wd * W + 0.2 * sqrt(max(m, n)) * Q)

    which effectively applies a weight-decay coefficient of ``lr * r_hat * wd``.
    The closest MuScale-family variant uses *decoupled* weight decay at full
    strength ``lr * wd``, independent of r_hat. With r held fixed at r != 1
    and wd > 0, the two updates must diverge.

    This test guards against accidentally regressing to decoupled weight decay.
    """
    lr = 0.02
    mu = 0.95
    wd = 0.1
    steps = 5
    r_fixed = 0.5

    w_coupled = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon_moonlight",
         "r_min": r_fixed, "r_max": r_fixed},
        steps=steps,
    )

    # Reference: MuScale forced to the same constant trust ratio r_fixed.
    # MuScale applies the same 0.2 * sqrt(max(m, n)) * Q * r_fixed term but
    # uses *decoupled* weight decay at full strength (lr * wd, unscaled by
    # r_fixed). The only difference between the two runs is therefore the
    # weight-decay coefficient, so they must diverge whenever wd > 0 and
    # r_fixed != 1.
    w_decoupled = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "muscale",
         "r_min": r_fixed, "r_max": r_fixed},
        steps=steps,
    )

    diff = (w_coupled.float() - w_decoupled.float()).norm()
    assert diff > 1e-3, \
        f"Expected coupled-WD OrScale-Muon-Moonlight to differ from decoupled reference, got diff = {diff:.6f}"


def test_orscale_muon_wd_constant_ratio_matches_muon_weight_decay():
    """With r=1, trust-scaled decay collapses to the standard Muon wd update."""
    lr = 0.02
    mu = 0.95
    wd = 0.1
    steps = 10

    w_orscale_wd = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon_wd", "r_min": 1.0, "r_max": 1.0},
        steps=steps,
    )

    w_muon = _get_weight_after_steps(
        Muon,
        {"lr": lr, "momentum": mu, "weight_decay": wd, "moonlight_rms": False},
        steps=steps,
    )

    diff = (w_orscale_wd.float() - w_muon.float()).norm()
    ref = w_muon.float().norm()
    rel_diff = (diff / ref).item()
    assert rel_diff < 0.01, \
        f"OrScale-Muon-WD(r=1) vs Muon(wd) relative diff = {rel_diff:.6f} (expected < 0.01)"


def test_orscale_muon_wd_differs_when_trust_ratio_scales_decay():
    """When r != 1 and wd > 0, the WD-scaled variant should differ from decoupled Muon."""
    lr = 0.02
    mu = 0.95
    wd = 0.1
    steps = 5

    w_orscale_muon = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon", "r_min": 0.5, "r_max": 0.5},
        steps=steps,
    )

    w_orscale_wd = _get_weight_after_steps(
        OrScaleOptimizer,
        {"lr": lr, "momentum": mu, "weight_decay": wd,
         "variant": "orscale_muon_wd", "r_min": 0.5, "r_max": 0.5},
        steps=steps,
    )

    diff = (w_orscale_muon.float() - w_orscale_wd.float()).norm()
    assert diff > 1e-3, \
        f"Expected trust-scaled WD variant to differ, but diff = {diff:.6f}"


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
