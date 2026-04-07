"""
Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

Reference: https://kellerjordan.github.io/posts/muon/

Muon applies SGD with Nesterov momentum, then orthogonalizes each 2D parameter's
update via Newton-Schulz iteration. This replaces the update with the nearest
orthogonal matrix, effectively performing steepest descent under the spectral norm.

This implementation supports:
- Nesterov momentum (canonical Muon)
- Decoupled weight decay
- Optional Moonlight RMS shape normalization (sqrt(max(m,n)))
- Diagnostic hooks for logging intermediate values
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from orscale.optim.newton_schulz import orthogonalize


class Muon(Optimizer):
    """
    Muon optimizer for 2D (matrix) parameters.

    Algorithm per parameter W of shape (m, n):
        1. G = grad(W)
        2. M_t = mu * M_{t-1} + G_t
        3. M_hat_t = mu * M_t + G_t          (Nesterov lookahead)
        4. Q_t = NS_5(M_hat_t)               (Newton-Schulz orthogonalization)
        5. W_{t+1} = (1 - lr*wd) * W_t - lr * scale * Q_t

    where scale = sqrt(max(m,n)) if moonlight_rms else 1.0.

    Args:
        params: Iterable of parameters to optimize (should be 2D).
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient mu (default: 0.95).
        weight_decay: Decoupled weight decay (default: 0.0).
        moonlight_rms: If True, apply Moonlight shape normalization
            by scaling the update by sqrt(max(m, n)) (default: False).
        ns_iters: Number of Newton-Schulz iterations (default: 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        moonlight_rms: bool = False,
        ns_iters: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            moonlight_rms=moonlight_rms,
            ns_iters=ns_iters,
        )
        super().__init__(params, defaults)
        self._diagnostics: dict[str, dict] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._diagnostics.clear()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            moonlight = group["moonlight_rms"]
            ns_iters = group["ns_iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim != 2:
                    # Fallback: plain SGD with momentum for non-2D params
                    # (Muon is only defined for matrices)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

                buf = state["momentum_buffer"]
                g = grad.float()

                # Nesterov momentum
                buf.mul_(mu).add_(g)
                m_hat = buf.mul(mu).add(g)

                # Newton-Schulz orthogonalization
                Q = orthogonalize(m_hat, num_iters=ns_iters)

                # Shape normalization (Moonlight)
                m, n = p.shape
                scale = math.sqrt(max(m, n)) if moonlight else 1.0

                # Decoupled weight decay + update
                p.mul_(1.0 - lr * wd)
                p.add_(Q.to(p.dtype), alpha=-lr * scale)

                # Expose diagnostics for the logger
                param_name = _get_param_name(p)
                if param_name:
                    self._diagnostics[param_name] = {
                        "W_frob": p.norm().item(),
                        "G_frob": grad.norm().item(),
                        "M_frob": buf.norm().item(),
                        "Q_frob": Q.norm().item(),
                        "W_rms": (p.norm() / math.sqrt(m * n)).item(),
                        "M_rms": (buf.norm() / math.sqrt(m * n)).item(),
                    }

        return loss


def _get_param_name(p: Tensor) -> Optional[str]:
    """Retrieve the diagnostic name attached to a parameter, if any."""
    return getattr(p, "_diag_name", None)
