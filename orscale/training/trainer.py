"""
Training loop for OrScale experiments.

Supports single-GPU and DDP multi-GPU training with bfloat16 autocast,
gradient accumulation, periodic validation, W&B logging, and checkpointing.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from orscale.diagnostics.logger import DiagnosticLogger
from orscale.training.scheduler import CosineWithWarmup
from orscale.utils.distributed import is_main_process, get_world_size, reduce_mean


class Trainer:
    """
    Main training loop for language model experiments.

    Args:
        model: GPT model instance.
        optimizers: Single optimizer or list of optimizers (e.g. [Muon, AdamW]).
        scheduler: CosineWithWarmup LR scheduler.
        train_loader: Iterable yielding (input_ids, targets) batches.
        val_loader: Iterable yielding (input_ids, targets) batches (optional).
        config: Training config dict with keys like max_steps, grad_accum_steps, etc.
        diagnostic_logger: Optional DiagnosticLogger instance.
        device: torch.device for training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: list,
        scheduler: CosineWithWarmup,
        train_loader,
        val_loader=None,
        config: dict[str, Any] | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wrap model in DDP if distributed
        self.model = model.to(self.device)
        if get_world_size() > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = DDP(self.model, device_ids=[local_rank])
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.diag = diagnostic_logger

        # Config
        self.max_steps = self.config.get("max_steps", 5000)
        self.grad_accum_steps = self.config.get("grad_accum_steps", 1)
        self.val_every = self.config.get("val_every", 250)
        self.val_steps = self.config.get("val_steps", 20)
        self.log_every = self.config.get("log_every", 10)
        self.save_every = self.config.get("save_every", 0)
        self.save_dir = self.config.get("save_dir", "checkpoints")
        self.use_amp = self.config.get("precision", "bfloat16") == "bfloat16"

        self._wandb = None
        if self.config.get("wandb_project") and is_main_process():
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

    def train(self):
        """Run the full training loop."""
        model = self.model
        model.train()

        train_iter = self._make_infinite_iter(self.train_loader)
        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        step = 0
        t0 = time.perf_counter()
        running_loss = 0.0
        tokens_seen = 0

        while step < self.max_steps:
            # --- LR schedule ---
            self.scheduler.step(step)

            # --- Accumulation loop ---
            total_loss = 0.0
            for micro_step in range(self.grad_accum_steps):
                input_ids, targets = next(train_iter)
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                    output = model(input_ids, targets)
                    loss = output["loss"] / self.grad_accum_steps

                loss.backward()
                total_loss += loss.item()

            # --- Optimizer step ---
            for opt in self.optimizers:
                opt.step()
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            step += 1
            running_loss += total_loss
            batch_tokens = input_ids.numel() * self.grad_accum_steps * get_world_size()
            tokens_seen += batch_tokens

            # --- Logging ---
            if is_main_process() and step % self.log_every == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = running_loss / self.log_every
                lr_mult = self.scheduler.get_last_lr(step)
                tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

                log_dict = {
                    "train/loss": avg_loss,
                    "train/lr_multiplier": lr_mult,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen": tokens_seen,
                    "train/step": step,
                }

                if self._wandb is not None:
                    self._wandb.log(log_dict, step=step)

                print(
                    f"step {step}/{self.max_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr_mult {lr_mult:.4f} | "
                    f"tok/s {tok_per_sec:.0f}"
                )
                running_loss = 0.0

            # --- Diagnostics ---
            if self.diag is not None:
                self.diag.collect(step)

            # --- Validation ---
            if self.val_loader is not None and self.val_every > 0 and step % self.val_every == 0:
                val_loss = self.validate()
                if is_main_process():
                    print(f"step {step}/{self.max_steps} | val_loss {val_loss:.4f}")
                    if self._wandb is not None:
                        self._wandb.log({"val/loss": val_loss}, step=step)

            # --- Checkpoint ---
            if self.save_every > 0 and step % self.save_every == 0 and is_main_process():
                self.save_checkpoint(step)

        # Final validation
        if self.val_loader is not None:
            val_loss = self.validate()
            if is_main_process():
                print(f"Final val_loss: {val_loss:.4f}")
                if self._wandb is not None:
                    self._wandb.log({"val/loss": val_loss}, step=step)

        if is_main_process():
            elapsed = time.perf_counter() - t0
            print(f"Training complete. {step} steps in {elapsed:.1f}s. "
                  f"Total tokens: {tokens_seen:,}")

        return step, tokens_seen

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return mean loss."""
        model = self.model
        model.eval()
        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        total_loss = 0.0
        count = 0
        val_iter = self._make_infinite_iter(self.val_loader)

        for _ in range(self.val_steps):
            input_ids, targets = next(val_iter)
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.use_amp):
                output = model(input_ids, targets)

            total_loss += output["loss"].item()
            count += 1

        mean_loss = total_loss / max(count, 1)

        # All-reduce across ranks
        loss_tensor = torch.tensor(mean_loss, device=self.device)
        reduce_mean(loss_tensor)

        model.train()
        return loss_tensor.item()

    def save_checkpoint(self, step: int):
        """Save model and optimizer state."""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"step_{step:06d}.pt")
        state = {
            "step": step,
            "model": self.raw_model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "config": self.config,
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the step number."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state["model"])
        for opt, opt_state in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(opt_state)
        print(f"Loaded checkpoint from {path} (step {state['step']})")
        return state["step"]

    @staticmethod
    def _make_infinite_iter(loader) -> Iterator:
        """Wrap a finite iterable into an infinite one by cycling."""
        while True:
            for batch in loader:
                yield batch
