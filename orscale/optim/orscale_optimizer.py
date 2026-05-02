"""
OrScale optimizer family: Orthogonalized updates with layer-wise trust-ratio scaling.

Combines Muon's orthogonalized update direction with several layer-wise trust
ratio designs in a single configurable class. Eight variants are implemented:

    OrScale-original                  : EMA momentum,  ||Q||_F denominator
    OrScale-Muon                      : Nesterov,      ||Q||_F denominator
    OrScale-Muon-WD                   : Nesterov,      ||λW + Q||_F denominator,
                                                       coupled WD update
    OrScale-Muon-Moonlight            : Nesterov,      ||λW + 0.2·√max(m,n)·Q||_F
                                                       denominator, 0.2·√max(m,n)
                                                       shape norm, coupled WD update
    OrScale-Muon-Moonlight-Calibrated : Nesterov,      c_denom_ℓ·||λW + 0.2·√max(m,n)·Q||_F
                                                       denominator (c_denom_ℓ
                                                       auto-calibrated per-layer
                                                       at step 0 so r̂_ℓ(0)=1),
                                                       0.2·√max(m,n) shape norm,
                                                       coupled WD update
    MuTrust                           : Nesterov,      ||M_hat||_F denominator
    MuScale                           : Nesterov,      RMS(M_hat) denominator,
                                                       0.2·√max(m,n) shape norm
    MuScale-alpha                     : Nesterov,      RMS(M_hat) denominator,
                                                       0.2·√max(m,n) shape norm,
                                                       partial exponent

All variants that apply Moonlight shape normalization use the full Moonlight
RMS-matching constant ``0.2·sqrt(max(m, n))`` (see ``MOONLIGHT_RMS_CONSTANT``).

The ``calibrated`` variant is a per-layer calibrated extension of
``orscale_muon_moonlight``: it shares the same "real update direction"
denominator ``||λW + 0.2·√max(m,n)·Q||_F`` and the same coupled-WD update,
but rescales the denominator by a per-layer constant ``c_denom_ℓ`` that
anchors the trust ratio so ``r̂_ℓ(0) = 1`` for every layer.  By default
``c_denom_ℓ`` is set per layer at step 0 from
``||W_ℓ(0)||_F / ||λW_ℓ(0) + 0.2·√max(m,n)·Q_ℓ(0)||_F``, which (i) makes the
trust ratio land at exactly 1 at initialization regardless of init scheme or
layer width, restoring muP-style learning-rate transferability, and (ii)
combined with coupled WD gives a stable fixed-point dynamic for ``r``: early
in training ``r ≈ ||W_t||/||W_0||`` provides LARS-style adaptation tracking
weight-norm growth, and asymptotically ``r → 1/(c_denom_ℓ · λ)`` is a finite
ceiling (the runaway feedback loop that broke the previous
``c_denom_ℓ · ||Q||_F`` decoupled-WD formulation is eliminated by coupled WD,
which scales the WD pull-back together with the Q step so ``r`` cannot drive
unbounded weight growth).

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
    ORSCALE_MUON_MOONLIGHT_CALIBRATED = "orscale_muon_moonlight_calibrated"
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
            'orscale_muon_moonlight', 'orscale_muon_moonlight_calibrated',
            'mutrust', 'muscale', 'muscale_alpha'.
        alpha: Trust ratio exponent. Only used for muscale_alpha (default: 0.5).
        r_min: Lower clipping bound for trust ratio (default: 0.5).  The
            recommended per-variant bounds are:

              * ``[0.5, 1.5]`` for the *non-Moonlight* variants
                (``orscale_muon``, ``orscale_muon_wd``, ``mutrust``,
                ``muscale``, ``muscale_alpha``, ``orscale_original``).  This
                tight clip was set on 2026-04-22 after the ``fineweb_bump``
                investigation showed that the original ``[0.1, 10.0]``
                bounds let some variants run at ~10x the nominal LR during
                warmup; see ``reports/fineweb_bump/`` and the sweep memo.
                The clip is the only thing keeping ``mutrust`` / ``muscale``
                from running at runaway effective LR (their raw ratio is
                ``O(1/lr)`` in practice; they saturate at ``r_max`` on
                ~100% of steps regardless of the chosen ``r_max``).

              * ``[0.1, 5.0]`` for the *Moonlight-shape* variants
                (``orscale_muon_moonlight``,
                ``orscale_muon_moonlight_calibrated``).  These have a
                shape-constant or auto-calibrated denominator with a wider
                natural operating range, and benefit from LARS/LAMB-style
                looser bounds.  The calibrated variant is auto-set so that
                ``r̂(0)=1`` per layer; the original Moonlight variant has
                ``r̂ ∈ [0.5, 0.74]`` at the optimal LR on FineWeb (16% of
                steps clipped at ``r_min=0.5`` under the tight bound), and
                ``[0.1, 5.0]`` removes that artefact while still catching
                pathological steps.

            The default value here (``0.5``) keeps the older / non-Moonlight
            variants at their tuned setting; sweep scripts in
            ``scripts/sweep_*.sh`` apply the looser ``[0.1, 5.0]`` bound
            via ``--set`` overrides for the Moonlight variants.
        r_max: Upper clipping bound for trust ratio (default: 1.5).  See
            ``r_min`` for the per-variant recommendation.
        eps: Numerical stability constant (default: 1e-6).
        ns_iters: Number of Newton-Schulz iterations (default: 5).
        c_denom: Per-layer reference scale used in the denominator of the
            ``orscale_muon_moonlight_calibrated`` variant only.  If ``None``
            (default), each layer's ``c_denom_ℓ`` is auto-calibrated at the
            first step from
            ``||W_ℓ(0)||_F / ||λW_ℓ(0) + 0.2·√max(m,n)·Q_ℓ(0)||_F``, which
            makes the trust ratio start at exactly 1 for every layer
            regardless of init scheme or layer width.  If a positive float
            is provided, the same value is used for every layer.  Ignored
            for all other variants.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        variant: str = "muscale_alpha",
        alpha: float = 0.5,
        r_min: float = 0.5,
        r_max: float = 1.5,
        eps: float = 1e-6,
        ns_iters: int = 5,
        c_denom: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if c_denom is not None and c_denom <= 0.0:
            raise ValueError(f"Invalid c_denom: {c_denom} (must be positive or None)")

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
            c_denom=c_denom,
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
            c_denom_user = group.get("c_denom", None)

            use_nesterov = variant != OrScaleVariant.ORSCALE_ORIGINAL
            use_rms = variant in (OrScaleVariant.MUSCALE, OrScaleVariant.MUSCALE_ALPHA)
            use_shape_norm = variant in (
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT,
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT_CALIBRATED,
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
            use_calibrated_denom = (
                variant == OrScaleVariant.ORSCALE_MUON_MOONLIGHT_CALIBRATED
            )
            scale_wd_with_trust = variant in (
                OrScaleVariant.ORSCALE_MUON_WD,
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT,
                OrScaleVariant.ORSCALE_MUON_MOONLIGHT_CALIBRATED,
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
                #
                # All trust-ratio arithmetic is done in float32 regardless of
                # the parameter / Q dtype.  ``orthogonalize`` returns Q in
                # bfloat16 (the NS iterations run in bfloat16 for speed); a
                # division in bfloat16 carries ~10^-3 relative error per
                # operation, which propagates directly into the update
                # magnitude.  LARS / LAMB reference implementations
                # (NVIDIA APEX, DeepSpeed) follow the same convention of
                # upcasting all norms to float32 before forming the trust
                # ratio.  ``m_hat`` is already float32 (constructed from the
                # float32 momentum buffer); the ``.float()`` calls on
                # ``p_view`` and ``Q`` are no-ops when the parameter is
                # already float32 and a cheap promotion otherwise.
                w_norm = p_view.norm().float()       # ||W||_F  (float32)
                q_norm = Q.norm().float()            # ||Q||_F  (float32)
                m_hat_norm = m_hat.norm().float()    # ||M_hat||_F  (float32)

                if use_rms:
                    numel = m * n
                    w_stat = w_norm / math.sqrt(numel)     # RMS(W)
                    m_stat = m_hat_norm / math.sqrt(numel) # RMS(M_hat)
                elif use_ortho_denom:
                    w_stat = w_norm
                    m_stat = q_norm                        # (degenerate)
                elif use_grad_matrix_denom:
                    w_stat = w_norm
                    # p_float is already float32; promote Q for an explicit
                    # float32 sum.  shape_scale and wd are Python floats.
                    grad_matrix = wd * p_float + shape_scale * Q.float()
                    m_stat = grad_matrix.norm()            # ||λW + s·Q||_F
                elif use_calibrated_denom:
                    # Coupled denominator c_denom_ℓ * ||λW + s·Q||_F.
                    #
                    # The denominator is the same "real update direction"
                    # ||λW + s·Q||_F used by ``orscale_muon_moonlight``,
                    # rescaled by a per-layer constant c_denom_ℓ that anchors
                    # r_ℓ(0) = 1.  c_denom_ℓ is set per layer at the first
                    # step from
                    #
                    #   c_denom_ℓ = ||W_ℓ(0)||_F / ||λW_ℓ(0) + s·Q_ℓ(0)||_F,
                    #
                    # giving r_ℓ(0) = 1 exactly for every layer regardless of
                    # init scheme or shape.  Combined with coupled WD (see the
                    # update step below), this gives a stable fixed-point
                    # dynamic: r ≈ ||W_t||_F / ||W_0||_F early in training
                    # (LARS-style adaptation), and asymptotically
                    # r → 1/(c_denom_ℓ · λ), bounded above (no runaway).  The
                    # earlier decoupled-WD ``c_denom_ℓ · ||Q||_F`` formulation
                    # ran ||W||_F up by 10–50× on FineWeb small_125m on three
                    # of four matrix-parameter classes; coupling the WD into
                    # the trust-ratio-scaled update damps that feedback loop.
                    grad_matrix = wd * p_float + shape_scale * Q.float()
                    grad_matrix_norm = grad_matrix.norm()  # ||λW + s·Q||_F
                    if "c_denom" not in state:
                        if c_denom_user is not None:
                            state["c_denom"] = float(c_denom_user)
                        else:
                            w_init = w_norm.item()
                            gm_init = grad_matrix_norm.item()
                            if w_init < eps or gm_init < eps:
                                # Fallback: pathological tiny init.  Use the
                                # Moonlight shape constant so the optimizer
                                # degenerates to scaled Moonlight rather than
                                # NaNing or producing a runaway ratio.
                                state["c_denom"] = (
                                    MOONLIGHT_RMS_CONSTANT * math.sqrt(max(m, n))
                                )
                            else:
                                state["c_denom"] = w_init / gm_init
                    c_denom_layer = state["c_denom"]
                    w_stat = w_norm
                    m_stat = c_denom_layer * grad_matrix_norm  # c_denom_ℓ · ||λW + s·Q||_F
                else:
                    # MuTrust: raw Frobenius norms
                    w_stat = w_norm
                    m_stat = m_hat_norm

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
                # Reuse the float32 norms computed above for the trust ratio
                # so the reported numbers match the values actually used in
                # the update (and so W&B does not log bfloat16-truncated
                # norms in mixed-precision training).
                param_name = getattr(p, "_diag_name", None)
                if param_name:
                    w_frob = w_norm.item()
                    q_frob = q_norm.item()
                    diag = {
                        "W_frob": w_frob,
                        "G_frob": grad.norm().float().item(),
                        "M_frob": buf_view.norm().float().item(),
                        "Q_frob": q_frob,
                        "W_rms": w_frob / math.sqrt(m * n),
                        "M_rms": buf_view.norm().float().item() / math.sqrt(m * n),
                        "trust_ratio_raw": ratio_raw_val,
                        "trust_ratio_clipped": ratio_clipped,
                        "clip_active": abs(ratio_clipped - ratio_raw_val) > 1e-8,
                        "shape_scale": shape_scale,
                        "weight_decay_scaled_by_trust": scale_wd_with_trust,
                        "update_to_param_ratio": (
                            lr * ratio_clipped * shape_scale * q_frob / (w_frob + eps)
                        ),
                    }
                    if use_calibrated_denom and "c_denom" in state:
                        diag["c_denom"] = state["c_denom"]
                    self._diagnostics[param_name] = diag

        return loss
