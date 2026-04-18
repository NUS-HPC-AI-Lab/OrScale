"""
Training loop for vision experiments (CIFAR-10 / ImageNet).

Mirrors ``orscale.training.trainer.Trainer`` but:
    - Uses an outer epoch loop (ImageNet schedules in LAMB are specified in epochs).
    - Computes top-1 / top-5 classification accuracy every eval.
    - Works with schedules that are either step-based (cosine, poly) or
      epoch-based (step decay).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from orscale.diagnostics.logger import DiagnosticLogger
from orscale.training.scheduler import (
    CosineWithWarmup,
    PolynomialDecayWithLinearWarmup,
    StepDecayWithWarmup,
)
from orscale.utils.distributed import is_main_process, get_world_size, reduce_mean


_STEP_BASED_SCHEDULERS = (CosineWithWarmup, PolynomialDecayWithLinearWarmup)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> list[float]:
    """Top-k accuracy averaged over the batch."""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k / batch_size).item())
    return results


class VisionTrainer:
    """
    Epoch-based training loop for image classification.

    Args:
        model: Vision model. Its forward should accept an image tensor and
            return logits of shape [batch, num_classes].
        optimizers: Single optimizer or list (e.g. [Muon, AdamW]).
        scheduler: Any scheduler from ``orscale.training.scheduler``.
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for validation.
        config: Dict with keys:
            - ``epochs``: number of epochs (int, default 24).
            - ``precision``: ``"bfloat16"`` or ``"fp32"`` (default ``"bfloat16"``).
            - ``grad_clip``: optional float; clip grad-norm if >0.
            - ``log_every``: log every N steps (default 50).
            - ``save_every_epoch``: save checkpoint every N epochs (default 0 = no save).
            - ``save_dir``: directory for checkpoints.
            - ``wandb_project``: if set and wandb installed, log there.
        diagnostic_logger: Optional ``DiagnosticLogger``.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers,
        scheduler,
        train_loader,
        val_loader,
        config: dict[str, Any] | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.epochs = int(self.config.get("epochs", 24))
        self.grad_clip = float(self.config.get("grad_clip", 0.0))
        self.log_every = int(self.config.get("log_every", 50))
        self.save_every_epoch = int(self.config.get("save_every_epoch", 0))
        self.save_dir = self.config.get("save_dir", "checkpoints")
        self.use_amp = self.config.get("precision", "bfloat16") == "bfloat16"
        self.label_smoothing = float(self.config.get("label_smoothing", 0.0))

        self._wandb = None
        if self.config.get("wandb_project") and is_main_process():
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                pass

        self._step_based = isinstance(self.scheduler, _STEP_BASED_SCHEDULERS)
        self._steps_per_epoch = len(self.train_loader)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        """Run the full training loop for ``self.epochs`` epochs."""
        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        t0 = time.perf_counter()
        history: list[dict[str, float]] = []

        for epoch in range(self.epochs):
            self._set_epoch(epoch)
            self.model.train()

            running = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "count": 0}
            for step_in_epoch, (images, labels) in enumerate(self.train_loader):
                fractional_epoch = epoch + step_in_epoch / max(1, self._steps_per_epoch)

                if self._step_based:
                    self.scheduler.step(self._global_step)
                else:
                    self.scheduler.step(fractional_epoch)

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.use_amp and self.device.type == "cuda"):
                    logits = self.model(images)
                    loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

                loss.backward()

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)

                for opt in self.optimizers:
                    opt.step()
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=True)

                # Running stats
                with torch.no_grad():
                    t1, t5 = accuracy(logits.detach().float(), labels)
                running["loss"] += loss.item()
                running["top1"] += t1
                running["top5"] += t5
                running["count"] += 1
                self._global_step += 1

                if is_main_process() and self._global_step % self.log_every == 0:
                    elapsed = time.perf_counter() - t0
                    lr_mult = (
                        self.scheduler.get_last_lr(self._global_step)
                        if self._step_based
                        else self.scheduler.get_last_lr(fractional_epoch)
                    )
                    print(
                        f"epoch {epoch}/{self.epochs} | "
                        f"step {self._global_step} | "
                        f"loss {running['loss'] / running['count']:.4f} | "
                        f"top1 {100 * running['top1'] / running['count']:.2f}% | "
                        f"lr_mult {lr_mult:.4f} | "
                        f"elapsed {elapsed:.1f}s"
                    )
                    if self._wandb is not None:
                        self._wandb.log(
                            {
                                "train/loss": running["loss"] / running["count"],
                                "train/top1": running["top1"] / running["count"],
                                "train/top5": running["top5"] / running["count"],
                                "train/lr_multiplier": lr_mult,
                                "train/epoch": fractional_epoch,
                            },
                            step=self._global_step,
                        )

                if self.diag is not None:
                    self.diag.collect(self._global_step)

            # End of epoch: validation
            val_metrics = self.validate()
            if is_main_process():
                print(
                    f"epoch {epoch}/{self.epochs} END | "
                    f"val_loss {val_metrics['loss']:.4f} | "
                    f"val_top1 {100 * val_metrics['top1']:.2f}% | "
                    f"val_top5 {100 * val_metrics['top5']:.2f}%"
                )
                if self._wandb is not None:
                    self._wandb.log(
                        {
                            "val/loss": val_metrics["loss"],
                            "val/top1": val_metrics["top1"],
                            "val/top5": val_metrics["top5"],
                            "val/epoch": epoch,
                        },
                        step=self._global_step,
                    )
            history.append({"epoch": epoch, **val_metrics})

            if (
                self.save_every_epoch > 0
                and (epoch + 1) % self.save_every_epoch == 0
                and is_main_process()
            ):
                self.save_checkpoint(epoch)

        if is_main_process():
            total = time.perf_counter() - t0
            print(f"Training complete. {self.epochs} epochs in {total:.1f}s.")
        return history

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        model = self.model
        model.eval()
        amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            bs = labels.size(0)
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.use_amp and self.device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            t1, t5 = accuracy(logits.float(), labels)
            total_loss += loss.item() * bs
            total_top1 += t1 * bs
            total_top5 += t5 * bs
            total_samples += bs

        if total_samples == 0:
            return {"loss": math.nan, "top1": 0.0, "top5": 0.0}

        metrics = {
            "loss": total_loss / total_samples,
            "top1": total_top1 / total_samples,
            "top5": total_top5 / total_samples,
        }

        # Aggregate across DDP ranks
        if get_world_size() > 1:
            for k in list(metrics.keys()):
                t = torch.tensor(metrics[k], device=self.device)
                reduce_mean(t)
                metrics[k] = t.item()

        model.train()
        return metrics

    def save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.pt")
        state = {
            "epoch": epoch,
            "step": self._global_step,
            "model": self.raw_model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "config": self.config,
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_epoch(self, epoch: int) -> None:
        """Seed DistributedSampler for deterministic shuffling across epochs."""
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
