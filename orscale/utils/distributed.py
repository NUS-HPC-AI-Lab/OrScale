"""
Distributed training utilities for DDP.

Provides helper functions wrapping torch.distributed for torchrun compatibility.
Gracefully handles single-GPU (non-distributed) operation.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, torch.device]:
    """
    Initialize distributed training if launched via torchrun / torch.distributed.launch.

    Returns:
        (rank, world_size, device) tuple.
        For single-GPU: (0, 1, cuda:0 or cpu).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        dist.init_process_group(backend="nccl")
        dist.barrier(device_ids=[local_rank])
        return rank, world_size, device

    # Single-process fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def cleanup_distributed():
    """Destroy the process group if distributed is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if this is rank 0 (or non-distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor by averaging across ranks. No-op if non-distributed."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor
