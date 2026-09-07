"""ZeRO-1 style optimizer-state sharding for replicated-parameter DDP.

This wrapper keeps model parameters and gradients replicated on every rank
(standard DDP semantics), but assigns each trainable parameter to exactly one
optimizer owner rank. Only the owner rank holds optimizer state and applies
the parameter update. After local updates, owners broadcast their updated
parameters to the other ranks.

That trade-off is deliberate for the Muon/OrScale family: full, unflattened
matrix gradients remain available for Newton-Schulz orthogonalization, while
the expensive momentum / Adam moments are sharded across the data-parallel
world. FSDP-style flattened parameters are not compatible with the current
matrix-wise optimizer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class Zero1Partition:
    """Deterministic owner assignment for a trainable parameter."""

    name: str
    param: nn.Parameter
    owner: int


def _dist_rank_world() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def make_zero1_partitions(model: nn.Module, world_size: int | None = None) -> list[Zero1Partition]:
    """Assign trainable parameters to owner ranks with a greedy size balance."""
    _, inferred_world = _dist_rank_world()
    world_size = int(world_size or inferred_world)
    if world_size < 1:
        raise ValueError("world_size must be positive")

    named_params = [
        (name, p)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    loads = [0] * world_size
    owners = [0] * len(named_params)

    # Largest-first bin packing gives much better balance for MoE models where
    # embeddings / LM heads and expert matrices are much larger than norms.
    order = sorted(
        range(len(named_params)),
        key=lambda i: named_params[i][1].numel(),
        reverse=True,
    )
    for idx in order:
        owner = min(range(world_size), key=lambda r: loads[r])
        owners[idx] = owner
        loads[owner] += named_params[idx][1].numel()

    return [
        Zero1Partition(name=name, param=param, owner=owner)
        for (name, param), owner in zip(named_params, owners)
    ]


def local_zero1_params(
    params: Iterable[nn.Parameter],
    partitions: list[Zero1Partition],
    rank: int | None = None,
) -> list[nn.Parameter]:
    """Filter ``params`` down to the subset owned by ``rank``."""
    rank, _ = _dist_rank_world() if rank is None else (int(rank), 1)
    owner_by_id = {id(part.param): part.owner for part in partitions}
    return [p for p in params if owner_by_id.get(id(p)) == rank]


class Zero1Optimizer:
    """A thin optimizer-like wrapper around rank-local optimizers."""

    is_zero1 = True

    def __init__(
        self,
        local_optimizers: list[torch.optim.Optimizer],
        partitions: list[Zero1Partition],
    ):
        self.local_optimizers = local_optimizers
        self.partitions = partitions
        self.rank, self.world_size = _dist_rank_world()
        self.param_groups = [
            group
            for opt in self.local_optimizers
            for group in opt.param_groups
        ]

    @property
    def state(self) -> dict:
        merged = {}
        for opt in self.local_optimizers:
            merged.update(opt.state)
        return merged

    @property
    def _diagnostics(self) -> dict:
        merged = {}
        for opt in self.local_optimizers:
            merged.update(getattr(opt, "_diagnostics", {}))
        return merged

    def step(self, closure=None):
        loss = None
        for opt in self.local_optimizers:
            maybe_loss = opt.step(closure=closure) if closure is not None else opt.step()
            if maybe_loss is not None:
                loss = maybe_loss
        self.sync_parameters()
        return loss

    @torch.no_grad()
    def sync_parameters(self) -> None:
        if not dist.is_initialized() or self.world_size <= 1:
            return
        for part in self.partitions:
            dist.broadcast(part.param.data, src=part.owner)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Clear every replicated parameter's grad, not just rank-local owned
        # params. Otherwise non-owner grads would accumulate forever.
        for part in self.partitions:
            grad = part.param.grad
            if grad is None:
                continue
            if set_to_none:
                part.param.grad = None
            else:
                grad.detach_()
                grad.zero_()

    def state_dict(self) -> dict:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_optimizers": [opt.state_dict() for opt in self.local_optimizers],
            "partitions": [
                {"name": part.name, "owner": part.owner}
                for part in self.partitions
            ],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        local_states = state_dict.get("local_optimizers", [])
        if len(local_states) != len(self.local_optimizers):
            raise ValueError(
                "Zero1 checkpoint optimizer count mismatch: "
                f"checkpoint has {len(local_states)}, runtime has {len(self.local_optimizers)}"
            )
        for opt, opt_state in zip(self.local_optimizers, local_states):
            opt.load_state_dict(opt_state)

    def local_state_numel(self) -> int:
        total = 0
        for opt in self.local_optimizers:
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        total += value.numel()
        return total
