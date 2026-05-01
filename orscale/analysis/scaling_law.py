"""
Scaling-law analysis utilities.

Reproduces the Moonlight (arXiv:2502.16982) Figure 3 / Table 3 workflow:

    1. Sweep (model size N, token budget D) pairs.
    2. Record final validation loss and compute cost (PFLOP/s-days).
    3. Fit a Chinchilla-style power law ``L(C) = A * C^alpha`` where C is the
       compute used (or ``L(N) = A * N^alpha`` etc.) via ``scipy.optimize.curve_fit``.
    4. Plot the Pareto frontier per optimizer.

This module only handles the post-training analysis. The sweep itself is
driven by ``scripts/run_scaling_law.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math


SECONDS_PER_DAY = 86_400
_PFLOPS = 1e15


# ---------------------------------------------------------------------------
# Compute accounting
# ---------------------------------------------------------------------------

def compute_flops(params: float, tokens: float, factor: float = 6.0) -> float:
    """Chinchilla-style FLOP estimate ``C ≈ factor * N * D``.

    Args:
        params: Number of non-embedding parameters ``N``.
        tokens: Number of training tokens ``D``.
        factor: Kaplan/Chinchilla coefficient (6 for forward+backward).

    Returns:
        Total training FLOPs (dimensionless, not divided by time).
    """
    return float(factor) * float(params) * float(tokens)


def compute_pflop_s_days(params: float, tokens: float, factor: float = 6.0) -> float:
    """Convert the ``6ND`` FLOP estimate to PFLOP/s-days for Kaplan-style plots.

    Kaplan et al. 2020 uses PF-days as the X-axis of the scaling-law plot::

        PF-days = FLOPs / 1e15 / 86400
    """
    flops = compute_flops(params, tokens, factor=factor)
    return flops / _PFLOPS / SECONDS_PER_DAY


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

@dataclass
class PowerLawFit:
    """Parameters of a fitted ``L(x) = A * x^alpha + offset`` law."""

    A: float
    alpha: float
    offset: float = 0.0

    def predict(self, x: float | Iterable[float]):
        import numpy as np
        x = np.asarray(x, dtype=float)
        return self.A * np.power(x, self.alpha) + self.offset


def fit_power_law(
    xs: Sequence[float],
    ys: Sequence[float],
    include_offset: bool = False,
    initial_alpha: float = -0.1,
) -> PowerLawFit:
    """Fit ``y = A * x^alpha (+ offset)`` via non-linear least squares.

    Use ``include_offset=True`` when fitting a loss floor (Chinchilla style),
    ``include_offset=False`` (the default) for Kaplan-style log-log fits.

    Args:
        xs: Independent variable (e.g. compute, tokens, parameters).
        ys: Dependent variable (e.g. final loss).
        include_offset: If True, fit ``A * x^alpha + offset``.
        initial_alpha: Initial guess for the exponent (should be negative
            for loss-vs-compute fits).

    Returns:
        A ``PowerLawFit`` dataclass with the fitted parameters.
    """
    try:
        import numpy as np
    except ImportError as err:
        raise ImportError("numpy is required for scaling-law fits. pip install numpy") from err

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape or xs.size < 2:
        raise ValueError("Need at least 2 matched (x, y) points to fit a power law.")

    if include_offset:
        try:
            from scipy.optimize import curve_fit

            def model(x, A, alpha, offset):
                return A * np.power(x, alpha) + offset

            p0 = (float(ys.max()), float(initial_alpha), float(ys.min() * 0.9))
            popt, _ = curve_fit(model, xs, ys, p0=p0, maxfev=20_000)
            return PowerLawFit(A=float(popt[0]), alpha=float(popt[1]), offset=float(popt[2]))
        except ImportError:
            return _fit_power_law_offset_numpy(xs, ys)

    # Log-log linear fit: log y = log A + alpha * log x.
    log_x = np.log(xs)
    log_y = np.log(ys)
    alpha, log_A = np.polyfit(log_x, log_y, 1)
    return PowerLawFit(A=float(math.exp(log_A)), alpha=float(alpha))


def _fit_power_law_offset_numpy(xs, ys) -> PowerLawFit:
    """Fit ``A * x^alpha + offset`` with a 1D NumPy-only offset search."""
    import numpy as np

    if np.any(ys <= 0):
        raise ValueError("Power-law losses must be positive.")

    upper = float(ys.min()) * 0.999
    lower = 0.0
    if upper <= lower:
        raise ValueError("Need positive losses to fit an offset power law.")

    def fit_at_offset(offset: float) -> tuple[float, float, float]:
        shifted = ys - offset
        if np.any(shifted <= 0):
            return float("inf"), 0.0, 0.0
        alpha, log_A = np.polyfit(np.log(xs), np.log(shifted), 1)
        A = float(math.exp(log_A))
        pred = A * np.power(xs, alpha) + offset
        sse = float(np.square(pred - ys).sum())
        return sse, A, float(alpha)

    # Golden-section search over the loss floor. This keeps the fallback small
    # and dependency-free while matching scipy's result for smooth loss curves.
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    a, b = lower, upper
    c = b - (b - a) * inv_phi
    d = a + (b - a) * inv_phi
    fc = fit_at_offset(c)[0]
    fd = fit_at_offset(d)[0]
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * inv_phi
            fc = fit_at_offset(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * inv_phi
            fd = fit_at_offset(d)[0]

    offset = (a + b) / 2.0
    _, A, alpha = fit_at_offset(offset)
    return PowerLawFit(A=A, alpha=alpha, offset=float(offset))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto(
    points_per_series: dict[str, Sequence[tuple[float, float]]],
    x_label: str = "Compute (PFLOP/s-days)",
    y_label: str = "Validation loss",
    fits: dict[str, PowerLawFit] | None = None,
    out_path: str | None = None,
    title: str | None = None,
) -> None:
    """Plot per-series scaling-law points and optional fitted curves.

    Args:
        points_per_series: ``{series_name: [(x, y), ...]}``.
        x_label, y_label: Axis labels.
        fits: Optional ``{series_name: PowerLawFit}`` to overlay.
        out_path: Save the figure to this path (PNG). If None, call ``plt.show()``.
        title: Optional figure title.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required to plot scaling laws. pip install matplotlib"
        ) from err

    fig, ax = plt.subplots(figsize=(7, 5))
    for series, points in points_per_series.items():
        if not points:
            continue
        xs, ys = zip(*sorted(points))
        ax.plot(xs, ys, "o", label=series)
        if fits and series in fits:
            grid = np.geomspace(min(xs), max(xs), 100)
            ax.plot(grid, fits[series].predict(grid), "--", alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
