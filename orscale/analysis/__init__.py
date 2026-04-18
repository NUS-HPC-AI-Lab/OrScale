"""Analysis utilities for OrScale (scaling laws, plotting)."""

from orscale.analysis.scaling_law import (
    compute_pflop_s_days,
    fit_power_law,
    plot_pareto,
)

__all__ = ["compute_pflop_s_days", "fit_power_law", "plot_pareto"]
