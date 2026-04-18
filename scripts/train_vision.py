#!/usr/bin/env python3
"""
Vision training entry point for OrScale experiments (CIFAR-10 / ImageNet).

Usage:
    Single GPU:
        python scripts/train_vision.py --config configs/cifar10_davidnet.yaml

    Multi-GPU (single node, DDP):
        torchrun --nproc_per_node=8 scripts/train_vision.py \
            --config configs/imagenet_resnet50_bs16k.yaml

    Multi-node (example: 8 nodes x 8 GPUs = 64 GPUs, total batch 16384):
        # On every node:
        torchrun \
            --nnodes=8 --nproc_per_node=8 \
            --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:29500 \
            scripts/train_vision.py \
            --config configs/imagenet_resnet50_bs16k.yaml

    Override config values:
        python scripts/train_vision.py \
            --config configs/cifar10_davidnet.yaml \
            --set optimizer.name=muon optimizer.lr=0.01 training.epochs=24
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orscale.data.vision import create_vision_dataloaders
from orscale.diagnostics.logger import DiagnosticLogger
from orscale.model.vision import build_vision_model
from orscale.optim import build_optimizer
from orscale.training.scheduler import build_scheduler, apply_sqrt_lr_scaling
from orscale.training.vision_trainer import VisionTrainer
from orscale.utils.distributed import (
    cleanup_distributed,
    get_world_size,
    is_main_process,
    setup_distributed,
)


LOGGER = logging.getLogger("orscale.train_vision")


def setup_terminal_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def log_main(message: str, *args) -> None:
    if is_main_process():
        LOGGER.info(message, *args)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    config = copy.deepcopy(config)
    for override in overrides:
        key, _, value = override.partition("=")
        if not value:
            raise ValueError(f"Invalid override: {override}. Expected key=value.")
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
            value = float(value) if "." in value or "e" in value.lower() else int(value)
        elif value.startswith("[") and value.endswith("]"):
            value = yaml.safe_load(value)
        d[keys[-1]] = value
    return config


def main():
    setup_terminal_logging()

    parser = argparse.ArgumentParser(description="OrScale vision training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    rank, world_size, device = setup_distributed()
    log_main("World size: %d, Device: %s", world_size, device)

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed + rank)

    # W&B init
    wandb_run = None
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("wandb_project") and is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project=logging_cfg["wandb_project"],
                group=logging_cfg.get("wandb_group"),
                name=logging_cfg.get("wandb_name"),
                config=config,
            )
        except ImportError:
            log_main("wandb not installed, skipping W&B logging.")

    # Model
    log_main("Building vision model: %s", config["model"]["name"])
    model = build_vision_model(config["model"])
    if is_main_process():
        pcount = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info("Model: %.2fM parameters", pcount / 1e6)

    # Optimizer with optional LAMB-style sqrt LR scaling
    opt_cfg = dict(config.get("optimizer", {}))
    per_rank_bs = int(train_cfg.get("batch_size", 128))
    global_bs = per_rank_bs * world_size
    if opt_cfg.get("sqrt_lr_scaling"):
        base_batch = int(opt_cfg.pop("sqrt_lr_scaling_base_batch", 512))
        scaled = apply_sqrt_lr_scaling(float(opt_cfg["lr"]), global_bs, base_batch)
        log_main(
            "Square-root LR scaling: %.4g -> %.4g (global_bs=%d, base=%d)",
            opt_cfg["lr"], scaled, global_bs, base_batch,
        )
        opt_cfg["lr"] = scaled
        opt_cfg.pop("sqrt_lr_scaling", None)

    optimizers = build_optimizer(opt_cfg.get("name", "adamw"), model, opt_cfg)
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    # Data
    data_cfg = dict(config.get("data", {}))
    log_main("Building data loaders: %s (per-rank bs=%d, global bs=%d)",
             data_cfg.get("name", "cifar10"), per_rank_bs, global_bs)
    train_loader, val_loader = create_vision_dataloaders(
        data_cfg, per_rank_bs, rank=rank, world_size=world_size, seed=seed,
    )

    # Scheduler
    sched_cfg = dict(config.get("scheduler", {}))
    sched_name = sched_cfg.pop("name", "cosine")
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(train_cfg.get("epochs", 24))
    sched_cfg.setdefault("max_steps", total_steps)
    sched_cfg.setdefault("total_steps", total_steps)
    scheduler = build_scheduler(sched_name, optimizers, sched_cfg)
    log_main(
        "Scheduler: %s (steps_per_epoch=%d, total_steps=%d)",
        sched_name, steps_per_epoch, total_steps,
    )

    # Diagnostic logger
    diag_cfg = config.get("diagnostics", {})
    diag_logger = DiagnosticLogger(
        model=model,
        optimizers=optimizers,
        log_every=diag_cfg.get("log_every", 50),
        heavy_log_every=diag_cfg.get("heavy_log_every", 500),
        use_wandb=wandb_run is not None,
    )

    trainer_config = {
        **train_cfg,
        "wandb_project": logging_cfg.get("wandb_project"),
    }
    trainer = VisionTrainer(
        model=model,
        optimizers=optimizers,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        diagnostic_logger=diag_logger,
        device=device,
    )

    log_main("Starting vision training loop (epochs=%d)...", trainer.epochs)
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
