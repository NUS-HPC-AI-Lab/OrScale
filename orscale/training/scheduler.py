"""
Learning rate schedulers for OrScale experiments.

Provides:
    - ``CosineWithWarmup``: linear warmup + cosine decay (default LM schedule).
    - ``PolynomialDecayWithLinearWarmup``: linear warmup + polynomial decay.
      Matches the LAMB BERT schedule ``eta_t = eta_0 * (1 - t/T)``
      (arXiv:1904.00962 Sec 4.1).
    - ``StepDecayWithWarmup``: linear-epoch warmup + step decay at milestones.
      Matches the Goyal et al. (2017) ImageNet schedule referenced in LAMB
      Tables 3/5 (multiply LR by ``gamma`` at epochs 30/60/80 with a 5-epoch
      warmup).
    - ``apply_sqrt_lr_scaling``: LAMB's square-root LR scaling rule for
      large-batch training (arXiv:1904.00962 Sec 4.3).
"""

from __future__ import annotations

import math
from typing import Iterable


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _BaseScheduler:
    """Base class that tracks per-group base LRs across a list of optimizers."""

    def __init__(self, optimizer):
        self.optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        self.base_lrs: list[list[float]] = []
        for opt in self.optimizers:
            self.base_lrs.append([g["lr"] for g in opt.param_groups])

    def _apply(self, mult: float) -> None:
        for opt, base_lr_list in zip(self.optimizers, self.base_lrs):
            for group, base_lr in zip(opt.param_groups, base_lr_list):
                group["lr"] = base_lr * mult


# ---------------------------------------------------------------------------
# Cosine with warmup (default LM schedule)
# ---------------------------------------------------------------------------

class CosineWithWarmup(_BaseScheduler):
    """Linear warmup followed by cosine decay.

    LR schedule:
        - Steps [0, warmup_steps):  linear ramp from 0 to base_lr
        - Steps [warmup_steps, max_steps]:  cosine decay to ``min_lr_ratio * base_lr``
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self, step: int) -> None:
        self._apply(self.get_lr_multiplier(step))

    def get_last_lr(self, step: int) -> float:
        return self.get_lr_multiplier(step)


# ---------------------------------------------------------------------------
# Polynomial decay with linear warmup (LAMB BERT schedule)
# ---------------------------------------------------------------------------

class PolynomialDecayWithLinearWarmup(_BaseScheduler):
    """Linear warmup followed by polynomial decay.

    Matches LAMB's BERT schedule (arXiv:1904.00962 Sec 4.1):
        ``eta_t = eta_0 * (1 - (t - warmup_steps) / (T - warmup_steps))^power``

    Args:
        optimizer: One or a list of optimizers.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (``T``).
        power: Polynomial decay exponent (default 1.0 = linear decay).
        min_lr_ratio: Final LR as fraction of peak (default 0.0).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        decay = (1.0 - progress) ** self.power
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay

    def step(self, step: int) -> None:
        self._apply(self.get_lr_multiplier(step))

    def get_last_lr(self, step: int) -> float:
        return self.get_lr_multiplier(step)


# ---------------------------------------------------------------------------
# Step decay with linear-epoch warmup (Goyal et al. ImageNet schedule)
# ---------------------------------------------------------------------------

class StepDecayWithWarmup(_BaseScheduler):
    """Linear-epoch warmup + step decay at fixed milestones.

    Matches the Goyal et al. (2017) ImageNet recipe used by LAMB Tables 3/5:
        - First ``warmup_epochs`` epochs: linear ramp from 0 to base_lr.
        - After each milestone (e.g. [30, 60, 80]): multiply LR by ``gamma``.

    Works in epoch units; call ``step(epoch)`` once per epoch (or with a
    fractional epoch for intra-epoch warmup: e.g. ``epoch + i/steps_per_epoch``).

    Args:
        optimizer: One or a list of optimizers.
        warmup_epochs: Linear warmup duration in epochs (can be fractional).
        milestones: Iterable of epochs at which to multiply LR by ``gamma``.
        gamma: Multiplicative decay factor (default 0.1).
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: float,
        milestones: Iterable[int] = (30, 60, 80),
        gamma: float = 0.1,
    ):
        super().__init__(optimizer)
        self.warmup_epochs = float(warmup_epochs)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)

    def get_lr_multiplier(self, epoch: float) -> float:
        if epoch < self.warmup_epochs:
            return epoch / max(1e-8, self.warmup_epochs)
        mult = 1.0
        for m in self.milestones:
            if epoch >= m:
                mult *= self.gamma
        return mult

    def step(self, epoch: float) -> None:
        self._apply(self.get_lr_multiplier(epoch))

    def get_last_lr(self, epoch: float) -> float:
        return self.get_lr_multiplier(epoch)


# ---------------------------------------------------------------------------
# LAMB-style square-root LR scaling rule
# ---------------------------------------------------------------------------

def apply_sqrt_lr_scaling(base_lr: float, batch_size: int, base_batch: int = 512) -> float:
    """Scale the base learning rate by ``sqrt(batch_size / base_batch)``.

    LAMB's recommended rule for large-batch training
    (arXiv:1904.00962 Sec 4.3, Tables 4/5): when you double the batch size,
    multiply the learning rate by ``sqrt(2)``.

    Args:
        base_lr: LR tuned at ``base_batch``.
        batch_size: The new (global) batch size.
        base_batch: The reference batch size ``base_lr`` was tuned at.

    Returns:
        The scaled learning rate.
    """
    if base_batch <= 0:
        raise ValueError("base_batch must be positive")
    return base_lr * math.sqrt(batch_size / base_batch)


def build_scheduler(name: str, optimizers, config: dict):
    """Dispatch to one of the scheduler classes above.

    ``config`` keys depend on ``name``:
        - ``cosine``: ``warmup_steps``, ``max_steps``, ``min_lr_ratio``.
        - ``poly``:   ``warmup_steps``, ``total_steps``, ``power``, ``min_lr_ratio``.
        - ``step``:   ``warmup_epochs``, ``milestones``, ``gamma``.
    """
    name = name.lower()
    if name == "cosine":
        return CosineWithWarmup(
            optimizers,
            warmup_steps=int(config.get("warmup_steps", 500)),
            max_steps=int(config.get("max_steps", 5000)),
            min_lr_ratio=float(config.get("min_lr_ratio", 0.0)),
        )
    if name in {"poly", "polynomial", "polynomial_decay"}:
        return PolynomialDecayWithLinearWarmup(
            optimizers,
            warmup_steps=int(config.get("warmup_steps", 500)),
            total_steps=int(config.get("total_steps", config.get("max_steps", 5000))),
            power=float(config.get("power", 1.0)),
            min_lr_ratio=float(config.get("min_lr_ratio", 0.0)),
        )
    if name in {"step", "step_decay"}:
        return StepDecayWithWarmup(
            optimizers,
            warmup_epochs=float(config.get("warmup_epochs", 5.0)),
            milestones=tuple(config.get("milestones", (30, 60, 80))),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unknown scheduler: {name!r}. Choose from cosine, poly, step.")
