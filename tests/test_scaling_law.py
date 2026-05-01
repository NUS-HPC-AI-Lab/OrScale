"""
Tests for the scaling-law analysis utilities.

Verifies that:

1. ``compute_pflop_s_days`` scales linearly in N and D with the expected
   conversion factor ``6ND / (1e15 * 86400)``.
2. ``fit_power_law`` recovers the exponent of a synthetic
   ``L = 3 * C^-0.05`` dataset to within 1% relative error.
3. ``fit_power_law(include_offset=True)`` can recover an offset.
4. ``plot_pareto`` writes a PNG file without errors.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from orscale.analysis.scaling_law import (
    compute_flops,
    compute_pflop_s_days,
    fit_power_law,
    plot_pareto,
)
from scripts.run_scaling_law import (
    derive_training_overrides,
    estimate_runtime_seconds,
    filter_by_name,
    format_duration,
    resolve_optimizer_lr,
    resolve_optimizer_value,
    split_filter_values,
)


def test_compute_flops_matches_6nd():
    assert compute_flops(1e9, 2e10) == pytest.approx(6 * 1e9 * 2e10)


def test_compute_pflop_s_days_units():
    # 6 * 1e9 * 2e10 = 1.2e20 FLOPs; / 1e15 = 1.2e5 PF; / 86400 = ~1.388 PF-days.
    expected = 6 * 1e9 * 2e10 / 1e15 / 86400
    assert compute_pflop_s_days(1e9, 2e10) == pytest.approx(expected)


def test_fit_power_law_recovers_exponent():
    """Synthetic L = A * C^alpha data should recover A, alpha within 1%."""
    A_true = 3.0
    alpha_true = -0.05
    xs = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    ys = [A_true * x ** alpha_true for x in xs]
    fit = fit_power_law(xs, ys, include_offset=False)
    assert fit.A == pytest.approx(A_true, rel=0.01)
    assert fit.alpha == pytest.approx(alpha_true, rel=0.01)
    assert fit.offset == 0.0


def test_fit_power_law_recovers_offset():
    """With include_offset=True, fit should recover L = A * C^alpha + offset."""
    A_true = 2.0
    alpha_true = -0.1
    offset_true = 1.5
    xs = [0.1, 1.0, 10.0, 100.0, 1000.0]
    ys = [A_true * x ** alpha_true + offset_true for x in xs]
    fit = fit_power_law(xs, ys, include_offset=True, initial_alpha=-0.1)
    assert fit.alpha == pytest.approx(alpha_true, rel=0.05)
    assert fit.offset == pytest.approx(offset_true, rel=0.05)


def test_fit_power_law_requires_two_points():
    with pytest.raises(ValueError):
        fit_power_law([1.0], [1.0])


def test_plot_pareto_writes_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    pts = {
        "adamw": [(0.1, 3.0), (1.0, 2.5), (10.0, 2.1)],
        "muon": [(0.1, 2.9), (1.0, 2.3), (10.0, 1.95)],
    }
    fits = {k: fit_power_law([x for x, _ in v], [y for _, y in v]) for k, v in pts.items()}
    out_path = tmp_path / "scaling_law.png"
    plot_pareto(pts, fits=fits, out_path=str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_strict_moonlight_batch_derivation():
    preset = {
        "name": "moonlight_399m",
        "tokens": 8.92e9,
        "seq_len": 8192,
        "batch_examples": 96,
    }
    overrides, meta = derive_training_overrides(
        preset,
        {"world_size": 8, "micro_batch_size": 1},
        {},
    )

    assert "model.max_seq_len=8192" in overrides
    assert "training.batch_size=1" in overrides
    assert "training.grad_accum_steps=12" in overrides
    assert "training.max_steps=11342" in overrides
    assert meta["tokens_per_step"] == 96 * 8192
    assert meta["actual_tokens"] == pytest.approx(11342 * 96 * 8192)


def test_strict_moonlight_batch_derivation_requires_divisible_batch():
    preset = {
        "name": "bad_batch",
        "tokens": 1e9,
        "seq_len": 8192,
        "batch_examples": 100,
    }

    with pytest.raises(ValueError, match="not divisible"):
        derive_training_overrides(preset, {"world_size": 8, "micro_batch_size": 3}, {})


def test_scaling_runner_filters_and_symbolic_lr():
    items = [{"name": "adamw"}, {"name": "muon_moonlight"}]
    selected = split_filter_values(["adamw,muon_moonlight"])
    assert filter_by_name(items, selected, kind="optimizer") == items

    preset = {"name": "moonlight_399m", "lr": 9.503e-4}
    optimizer = {"name": "muon_moonlight", "adamw_lr": "same_as_lr"}
    lr = resolve_optimizer_lr(preset, optimizer)

    assert lr == pytest.approx(9.503e-4)
    assert resolve_optimizer_value(optimizer["adamw_lr"], lr=lr) == pytest.approx(lr)


def test_runtime_estimate_uses_effective_pflops():
    seconds = estimate_runtime_seconds(
        params=1.25e8,
        actual_tokens=2.62144e9,
        cfg={"estimate_pflops_per_second": 0.5},
        preset={},
    )

    assert seconds == pytest.approx(6 * 1.25e8 * 2.62144e9 / 0.5e15)
    assert format_duration(seconds) == "1h 5m"
