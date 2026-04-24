"""
Optimizer implementations for the OrScale project.

Provides a factory function ``build_optimizer`` that constructs the appropriate
optimizer(s) for a model, automatically splitting parameters into matrix (2D)
and non-matrix groups.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from orscale.optim.lamb import LAMB
from orscale.optim.muon import Muon
from orscale.optim.orscale_optimizer import OrScaleOptimizer, OrScaleVariant

__all__ = [
    "Muon",
    "OrScaleOptimizer",
    "OrScaleVariant",
    "LAMB",
    "build_optimizer",
]

# Optimizer names that use the Muon-family (orthogonalized updates for 2D params)
_MUON_FAMILY = {
    "muon",
    "muon_moonlight",
    "orscale_original",
    "orscale_muon",
    "orscale_muon_wd",
    "orscale_muon_moonlight",
    "mutrust",
    "muscale",
    "muscale_alpha",
}


def _split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into matrix and non-matrix groups.

    Matrix params = weight tensors that can be orthogonalized by the Muon
    family. This includes:
      - Linear weights (ndim == 2) that are not embeddings.
      - Conv2d/Conv1d weights (ndim == 3 or 4) flattened internally by the
        optimizer (see Keller Jordan's blog: conv weights are treated as
        2D by flattening the last input dims).

    Non-matrix params = everything else (embeddings, norms, biases, 1D).

    Uses the ``muon_class`` attribute if set on a parameter; otherwise infers
    from shape and parent module type.
    """
    matrix_params = []
    nonmatrix_params = []

    # Build a set of embedding weight ids for exclusion
    embedding_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            embedding_ids.add(id(m.weight))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        explicit = getattr(p, "muon_class", None)
        if explicit == "matrix":
            matrix_params.append(p)
        elif explicit == "nonmatrix":
            nonmatrix_params.append(p)
        elif p.ndim >= 2 and id(p) not in embedding_ids:
            matrix_params.append(p)
        else:
            nonmatrix_params.append(p)

    return matrix_params, nonmatrix_params


def build_optimizer(
    name: str,
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
    """
    Build optimizer(s) for a model based on a config dict.

    For Muon-family optimizers, returns a list of two optimizers:
        [muon_optimizer (for 2D params), adamw_optimizer (for non-2D params)]

    For standard optimizers (adamw, lamb), returns a single optimizer.

    Args:
        name: Optimizer name. One of 'adamw', 'lamb', 'muon', 'muon_moonlight',
              'orscale_original', 'orscale_muon', 'orscale_muon_wd',
              'orscale_muon_moonlight', 'mutrust', 'muscale', 'muscale_alpha'.
        model: The nn.Module to optimize.
        config: Dict of optimizer hyperparameters. Keys depend on the optimizer.
            Common keys: lr, weight_decay.
            Muon-family keys: momentum, ns_iters.
            OrScale keys: alpha, r_min, r_max, variant.
            AdamW fallback keys: adamw_lr, adamw_betas, adamw_weight_decay, adamw_eps.

    Returns:
        A single optimizer or a list [matrix_opt, nonmatrix_opt].
    """
    name = name.lower().strip()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-8),
            weight_decay=config.get("weight_decay", 0.01),
        )

    if name == "lamb":
        return LAMB(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            betas=tuple(config.get("betas", (0.9, 0.999))),
            eps=config.get("eps", 1e-6),
            weight_decay=config.get("weight_decay", 0.01),
            clamp_value=config.get("clamp_value", 10.0),
        )

    if name not in _MUON_FAMILY:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Choose from: adamw, lamb, {', '.join(sorted(_MUON_FAMILY))}"
        )

    # Muon-family: split params into matrix / non-matrix
    matrix_params, nonmatrix_params = _split_params(model)

    if not matrix_params:
        raise ValueError(
            f"No 2D matrix parameters found for {name}. "
            "Muon-family optimizers require at least one 2D parameter."
        )

    # Build AdamW for non-matrix params
    adamw_opt = None
    if nonmatrix_params:
        adamw_opt = torch.optim.AdamW(
            nonmatrix_params,
            lr=config.get("adamw_lr", config.get("lr", 1e-3) * 0.1),
            betas=tuple(config.get("adamw_betas", (0.9, 0.999))),
            eps=config.get("adamw_eps", 1e-8),
            weight_decay=config.get("adamw_weight_decay", config.get("weight_decay", 0.01)),
        )

    # Build the Muon-family optimizer for matrix params
    if name == "muon":
        matrix_opt = Muon(
            matrix_params,
            lr=config.get("lr", 0.02),
            momentum=config.get("momentum", 0.95),
            weight_decay=config.get("weight_decay", 0.0),
            moonlight_rms=False,
            ns_iters=config.get("ns_iters", 5),
        )
    elif name == "muon_moonlight":
        matrix_opt = Muon(
            matrix_params,
            lr=config.get("lr", 0.02),
            momentum=config.get("momentum", 0.95),
            weight_decay=config.get("weight_decay", 0.0),
            moonlight_rms=True,
            ns_iters=config.get("ns_iters", 5),
        )
    else:
        # OrScale variants
        variant = name if name in {
            "orscale_original",
            "orscale_muon",
            "orscale_muon_wd",
            "orscale_muon_moonlight",
            "mutrust",
            "muscale",
            "muscale_alpha",
        } else config.get("variant", name)
        matrix_opt = OrScaleOptimizer(
            matrix_params,
            lr=config.get("lr", 0.02),
            momentum=config.get("momentum", 0.95),
            weight_decay=config.get("weight_decay", 0.0),
            variant=variant,
            alpha=config.get("alpha", 0.5),
            r_min=config.get("r_min", 0.5),
            r_max=config.get("r_max", 1.5),
            eps=config.get("eps", 1e-6),
            ns_iters=config.get("ns_iters", 5),
        )

    if adamw_opt is not None:
        return [matrix_opt, adamw_opt]
    return [matrix_opt]
