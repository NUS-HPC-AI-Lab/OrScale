"""
OrScale optimizer family: Orthogonalized updates with layer-wise trust-ratio scaling.

Combines Muon's orthogonalized update direction with several layer-wise trust
ratio designs in a single configurable class. Seven variants are implemented:

    OrScale-original       : EMA momentum,  ||Q||_F denominator
    OrScale-Muon           : Nesterov,      ||Q||_F denominator
    OrScale-Muon-WD        : Nesterov,      ||λW + Q||_F denominator, coupled WD update
    OrScale-Muon-Moonlight : Nesterov,      ||λW + 0.2·√max(m,n)·Q||_F denominator,
                                            0.2·√max(m,n) shape norm, coupled WD update
    MuTrust                : Nesterov,      ||M_hat||_F denominator
    MuScale                : Nesterov,      RMS(M_hat) denominator, 0.2·√max(m,n) shape norm
    MuScale-alpha          : Nesterov,      RMS(M_hat) denominator, 0.2·√max(m,n) shape norm,
                                            partial exponent

All variants that apply Moonlight shape normalization use the full Moonlight
RMS-matching constant ``0.2·sqrt(max(m, n))`` (see ``MOONLIGHT_RMS_CONSTANT``).

See the OrScale research memo for the full derivation and motivation.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from orscale.optim.muon import MOONLIGHT_RMS_CONSTANT
from orscale.optim.newton_schulz import orthogonalize


class OrScaleVariant(str, Enum):
    ORSCALE_ORIGINAL = "orscale_original"
    ORSCALE_MUON = "orscale_muon"
    ORSCALE_MUON_WD = "orscale_muon_wd"
    ORSCALE_MUON_MOONLIGHT = "orscale_muon_moonlight"
    MUTRUST = "mutrust"
    MUSCALE = "muscale"
    MUSCALE_ALPHA = "muscale_alpha"


class OrScaleOptimizer(Optimizer):
    """
    Unified OrScale optimizer implementing all supported variants.

    General algorithm for each 2D parameter W of shape (m, n):

        1. G = grad(W)
        2. M_t = momentum_update(M_{t-1}, G_t)   [EMA or accumulating]
        3. M_hat_t = nesterov_lookahead(M_t, G_t)  [skip for EMA variant]
        4. Q_t = NS_k(M_hat_t)                     [orthogonalization]
        5. r_t = (num(W) / (den(M_hat) + eps))^alpha  [trust ratio]
        6. r_hat_t = clip(r_t, r_min, r_max)
        7. Apply either:
           a) decoupled decay: W_{t+1} = (1 - lr*wd)*W_t - lr * r_hat_t * shape_scale * Q_t
           b) trust-scaled decay: W_{t+1} = W_t - lr * r_hat_t * (wd * W_t + shape_scale * Q_t)

    The variant determines the choice of numerator/denominator norms, momentum
    type, shape normalization, and exponent.

    Args:
        params: Iterable of 2D parameters.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Decoupled weight decay (default: 0.0).
        variant: One of 'orscale_original', 'orscale_muon', 'orscale_muon_wd',
            'orscale_muon_moonlight', 'mutrust', 'muscale', 'muscale_alpha'.
        alpha: Trust ratio exponent. Only used for muscale_alpha (default: 0.5).
        r_min: Lower clipping bound for trust ratio (default: 0.1).
        r_max: Upper clipping bound for trust ratio (default: 10.0).
        eps: Numerical stability constant (default: 1e-6).
        ns_iters: Number of Newton-Schulz iterations (default: 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        variant: str = "muscale_alpha",
        alpha: float = 0.5,
        r_min: float = 0.1,
        r_max: float = 10.0,
        eps: float = 1e-6,
        ns_iters: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        variant_enum = OrScaleVariant(variant)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            variant=variant_enum,
            alpha=alpha,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
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
            variant = group["variant"]
            alpha = group["alpha"]
            r_min = group["r_min"]
            r_max = group["r_max"]
            eps = group["eps"]
            ns_iters = group["ns_iters"]

            use_nesterov = variant != OrScaleVariant.ORSCALE_ORIGINAL
            use_rms = variant in (OrScaleVariant.MUSCALE, OrScaleVariant.MUSCALE_ALPHA)
            use_shape_norm = variant in (
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT,
                OrScaleVariant.MUSCALE,
                OrScaleVariant.MUSCALE_ALPHA,
            )
            use_ortho_denom = variant in (
                OrScaleVariant.ORSCALE_ORIGINAL,
                OrScaleVariant.ORSCALE_MUON,
            )
            use_grad_matrix_denom = variant in (
                OrScaleVariant.ORSCALE_MUON_WD,
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT,
            )
            scale_wd_with_trust = variant in (
                OrScaleVariant.ORSCALE_MUON_WD,
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT,
            )
            effective_alpha = alpha if variant == OrScaleVariant.MUSCALE_ALPHA else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim < 2:
                    continue

                orig_shape = p.shape
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

                buf = state["momentum_buffer"]

                # Flatten >=3D Conv weights to 2D [C_out, C_in*kH*kW]
                g = grad.float().reshape(orig_shape[0], -1)
                buf_view = buf.view(orig_shape[0], -1)
                p_view = p.view(orig_shape[0], -1)
                p_float = p_view.float()
                m, n = p_view.shape

                # --- Momentum ---
                if use_nesterov:
                    # Accumulating momentum: M_t = mu * M_{t-1} + G_t
                    buf_view.mul_(mu).add_(g)
                    # Nesterov lookahead: M_hat = mu * M_t + G_t
                    m_hat = buf_view.mul(mu).add(g)
                else:
                    # EMA momentum: M_t = beta * M_{t-1} + (1-beta) * G_t
                    buf_view.mul_(mu).add_(g, alpha=1.0 - mu)
                    m_hat = buf_view

                # --- Orthogonalization ---
                Q = orthogonalize(m_hat, num_iters=ns_iters)

                # --- Shape normalization ---
                # Every variant that opts into shape normalization uses the full
                # Moonlight RMS-matching scale 0.2 * sqrt(max(m, n)) (arXiv:2502.16982).
                shape_scale = (
                    MOONLIGHT_RMS_CONSTANT * math.sqrt(max(m, n))
                    if use_shape_norm
                    else 1.0
                )

                # --- Trust ratio ---
                if use_rms:
                    numel = m * n
                    w_stat = p_view.norm() / math.sqrt(numel)     # RMS(W)
                    m_stat = m_hat.norm() / math.sqrt(numel)      # RMS(M_hat)
                elif use_ortho_denom:
                    w_stat = p_view.norm()   # ||W||_F
                    m_stat = Q.norm()        # ||Q||_F (degenerate)
                elif use_grad_matrix_denom:
                    w_stat = p_view.norm()   # ||W||_F
                    grad_matrix = wd * p_float + shape_scale * Q
                    m_stat = grad_matrix.norm()  # ||λW + shape_scale·Q||_F
                else:
                    # MuTrust: Frobenius norms
                    w_stat = p_view.norm()    # ||W||_F
                    m_stat = m_hat.norm()     # ||M_hat||_F

                ratio_raw = (w_stat / (m_stat + eps)) ** effective_alpha
                ratio_raw_val = ratio_raw.item() if isinstance(ratio_raw, Tensor) else float(ratio_raw)
                ratio_clipped = max(r_min, min(r_max, ratio_raw_val))

                # --- Update ---
                q_update = Q.to(p.dtype)
                if scale_wd_with_trust:
                    if wd != 0.0:
                        p_view.add_(p_view, alpha=-lr * ratio_clipped * wd)
                    p_view.add_(q_update, alpha=-lr * ratio_clipped * shape_scale)
                else:
                    p_view.mul_(1.0 - lr * wd)
                    p_view.add_(q_update, alpha=-lr * ratio_clipped * shape_scale)

                # --- Diagnostics ---
                param_name = getattr(p, "_diag_name", None)
                if param_name:
                    self._diagnostics[param_name] = {
                        "W_frob": p_view.norm().item(),
                        "G_frob": grad.norm().item(),
                        "M_frob": buf_view.norm().item(),
                        "Q_frob": Q.norm().item(),
                        "W_rms": (p_view.norm() / math.sqrt(m * n)).item(),
                        "M_rms": (buf_view.norm() / math.sqrt(m * n)).item(),
                        "trust_ratio_raw": ratio_raw_val,
                        "trust_ratio_clipped": ratio_clipped,
                        "clip_active": abs(ratio_clipped - ratio_raw_val) > 1e-8,
                        "shape_scale": shape_scale,
                        "weight_decay_scaled_by_trust": scale_wd_with_trust,
                        "update_to_param_ratio": (
                            lr * ratio_clipped * shape_scale * Q.norm() / (p_view.norm() + eps)
                        ).item(),
                    }

        return loss
