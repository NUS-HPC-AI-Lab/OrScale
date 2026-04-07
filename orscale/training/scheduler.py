"""
Learning rate schedulers for OrScale experiments.

CosineWithWarmup: Linear warmup followed by cosine decay. Standard schedule
for optimizer comparison papers.
"""

from __future__ import annotations

import math


class CosineWithWarmup:
    """
    Cosine annealing with linear warmup.

    LR schedule:
        - Steps [0, warmup_steps):  linear ramp from 0 to base_lr
        - Steps [warmup_steps, max_steps]:  cosine decay from base_lr to min_lr

    Args:
        optimizer: A torch.optim.Optimizer or list of optimizers.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Final LR as a fraction of peak LR (default: 0.0).
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.0,
    ):
        self.optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        # Store base LRs from each optimizer's param groups
        self.base_lrs: list[list[float]] = []
        for opt in self.optimizers:
            self.base_lrs.append([g["lr"] for g in opt.param_groups])

    def get_lr_multiplier(self, step: int) -> float:
        """Return the LR multiplier for the given step."""
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        if step >= self.max_steps:
            return self.min_lr_ratio

        # Cosine decay phase
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self, step: int):
        """Update all optimizer learning rates for the given step."""
        mult = self.get_lr_multiplier(step)
        for opt, base_lr_list in zip(self.optimizers, self.base_lrs):
            for group, base_lr in zip(opt.param_groups, base_lr_list):
                group["lr"] = base_lr * mult

    def get_last_lr(self, step: int) -> float:
        """Return the current LR multiplier (convenience for logging)."""
        return self.get_lr_multiplier(step)
