"""
DiagnosticLogger for OrScale optimizers.

Collects per-layer metrics from Muon-family optimizers at configurable
intervals and logs them to Weights & Biases (if available) or a local dict.

Light metrics (norms, trust ratios) are collected every ``log_every`` steps.
Heavy metrics (singular values, QK logit stats) every ``heavy_log_every`` steps.

In addition to per-layer keys (``diagnostics/<param>/<metric>``), a small set
of aggregate keys (``diagnostics/_summary/<metric>_{mean,min,max,active_frac}``)
is emitted so that training dashboards can track the most useful signals --
trust-ratio clipping activity, update-to-param ratio, etc. -- with a single
line rather than one per layer.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


# Metric names for which we also emit cross-layer aggregate summaries
# (mean / min / max). Booleans are additionally summarized as a
# "<name>_active_frac" fraction.
_AGGREGATE_METRICS: tuple[str, ...] = (
    "trust_ratio_raw",
    "trust_ratio_clipped",
    "clip_active",
    "update_to_param_ratio",
    "W_rms",
    "M_rms",
    "shape_scale",
    "weight_decay_scaled_by_trust",
)


class DiagnosticLogger:
    """
    Collects and logs per-layer optimizer diagnostics.

    Reads intermediate values from the optimizer's ``_diagnostics`` dict,
    which is populated during each optimizer step. Also hooks into model
    forward passes to collect attention statistics.

    Args:
        model: The nn.Module being trained.
        optimizers: List of optimizers (one or more). Must have ``_diagnostics``.
        log_every: Steps between light metric collection (default: 50).
        heavy_log_every: Steps between heavy metric collection (default: 500).
        use_wandb: If True, log to W&B. If False, store in ``self.history``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: list,
        log_every: int = 50,
        heavy_log_every: int = 500,
        use_wandb: bool = True,
    ):
        self.model = model
        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        self.log_every = log_every
        self.heavy_log_every = heavy_log_every
        self.use_wandb = use_wandb
        self.history: list[dict[str, Any]] = []

        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self.use_wandb = False

    def should_log(self, step: int) -> bool:
        return step > 0 and step % self.log_every == 0

    def should_heavy_log(self, step: int) -> bool:
        return step > 0 and step % self.heavy_log_every == 0

    @torch.no_grad()
    def collect(self, step: int) -> dict[str, Any] | None:
        """
        Collect diagnostics for the current step.

        Returns the metrics dict if this is a logging step, else None.
        """
        if not self.should_log(step):
            return None

        metrics: dict[str, Any] = {"step": step}
        heavy = self.should_heavy_log(step)

        # Collect optimizer diagnostics. We also accumulate per-metric buckets
        # so we can emit cross-layer aggregates below.
        buckets: dict[str, list[float]] = {}
        for opt in self.optimizers:
            diag = getattr(opt, "_diagnostics", {})
            for param_name, param_diag in diag.items():
                for metric_name, value in param_diag.items():
                    key = f"diagnostics/{param_name}/{metric_name}"
                    metrics[key] = value
                    if metric_name in _AGGREGATE_METRICS:
                        try:
                            buckets.setdefault(metric_name, []).append(float(value))
                        except (TypeError, ValueError):
                            pass

        # Cross-layer aggregates: one value per metric, easy to plot.
        for metric_name, values in buckets.items():
            if not values:
                continue
            prefix = f"diagnostics/_summary/{metric_name}"
            metrics[f"{prefix}_mean"] = sum(values) / len(values)
            metrics[f"{prefix}_min"] = min(values)
            metrics[f"{prefix}_max"] = max(values)
            # For booleans (clip_active, weight_decay_scaled_by_trust) the
            # fraction of layers where the flag is set is the headline number.
            if all(v in (0.0, 1.0) for v in values):
                metrics[f"{prefix}_active_frac"] = (
                    sum(values) / len(values)
                )
            # For a scalar like update_to_param_ratio, the max across layers
            # is the canonical early-warning signal for instability.

        # Heavy metrics: singular values of momentum / orthogonalized updates
        if heavy:
            metrics.update(self._collect_heavy_metrics())

        self.history.append(metrics)

        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

        return metrics

    def _collect_heavy_metrics(self) -> dict[str, Any]:
        """Collect expensive metrics: top singular values of optimizer buffers."""
        metrics = {}

        for opt in self.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    name = getattr(p, "_diag_name", None)
                    if name is None or p.ndim != 2:
                        continue

                    state = opt.state.get(p, {})
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        continue

                    # Top-5 singular values of momentum buffer
                    try:
                        k = min(5, min(buf.shape))
                        svs = torch.linalg.svdvals(buf.float())[:k]
                        for i, sv in enumerate(svs):
                            metrics[f"diagnostics/{name}/sv_M_{i}"] = sv.item()
                    except Exception:
                        pass

        return metrics

    def get_summary(self) -> dict[str, list[float]]:
        """Return a dict mapping metric keys to lists of values over time."""
        summary: dict[str, list[float]] = {}
        for entry in self.history:
            for k, v in entry.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float)):
                    summary.setdefault(k, []).append(v)
        return summary
