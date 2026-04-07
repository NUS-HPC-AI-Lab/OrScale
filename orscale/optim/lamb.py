"""
LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training).

Reference: You et al., "Large Batch Optimization for Deep Learning: Training BERT
in 76 Minutes" (2019). https://arxiv.org/abs/1904.00962

LAMB computes an Adam-style update direction and then applies a layer-wise trust
ratio ||W||_F / ||update||_F to scale the step. This is included as a baseline
for the OrScale project.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class LAMB(Optimizer):
    """
    LAMB optimizer.

    Algorithm per parameter W:
        1. g = grad(W)
        2. m_t = beta1 * m_{t-1} + (1 - beta1) * g          (first moment)
        3. v_t = beta2 * v_{t-1} + (1 - beta2) * g^2        (second moment)
        4. m_hat = m_t / (1 - beta1^t)                       (bias correction)
        5. v_hat = v_t / (1 - beta2^t)
        6. u_t = m_hat / (sqrt(v_hat) + eps) + wd * W        (Adam direction + WD)
        7. r_t = ||W||_F / ||u_t||_F                         (trust ratio)
        8. W_{t+1} = W_t - lr * r_t * u_t

    Args:
        params: Iterable of parameters.
        lr: Learning rate (default: 1e-3).
        betas: Adam momentum coefficients (default: (0.9, 0.999)).
        eps: Numerical stability (default: 1e-6).
        weight_decay: Weight decay coefficient (default: 0.01).
        clamp_value: Upper bound for trust ratio (default: 10.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        clamp_value: float = 10.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_value=clamp_value,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            clamp_value = group["clamp_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Adam update direction + weight decay
                update = m_hat / (v_hat.sqrt() + eps)
                if wd > 0:
                    update.add_(p.float(), alpha=wd)

                # Layer-wise trust ratio
                w_norm = p.float().norm()
                u_norm = update.norm()

                if w_norm > 0 and u_norm > 0:
                    trust_ratio = (w_norm / u_norm).clamp(max=clamp_value).item()
                else:
                    trust_ratio = 1.0

                p.add_(update.to(p.dtype), alpha=-lr * trust_ratio)

        return loss
