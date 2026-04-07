#!/usr/bin/env python3
"""
Main training entry point for OrScale experiments.

Usage:
    Single GPU:
        python scripts/train.py --config configs/pilot_25m.yaml

    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 scripts/train.py --config configs/pilot_25m.yaml

    Override config values:
        python scripts/train.py --config configs/pilot_25m.yaml \
            --set optimizer.name=muon optimizer.lr=0.01 training.max_steps=2000
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orscale.model.gpt import GPT, GPTConfig, PRESET_CONFIGS
from orscale.optim import build_optimizer
from orscale.data.loader import create_dataloader
from orscale.diagnostics.logger import DiagnosticLogger
from orscale.training.trainer import Trainer
from orscale.training.scheduler import CosineWithWarmup
from orscale.utils.distributed import setup_distributed, cleanup_distributed, is_main_process


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dot-separated key=value overrides to a nested config dict."""
    config = copy.deepcopy(config)
    for override in overrides:
        key, _, value = override.partition("=")
        if not value:
            raise ValueError(f"Invalid override: {override}. Expected key=value.")

        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})

        # Auto-cast types
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
            value = float(value) if "." in value or "e" in value.lower() else int(value)
        elif value.startswith("[") and value.endswith("]"):
            value = yaml.safe_load(value)

        d[keys[-1]] = value
    return config


def build_model(model_config: dict, device: torch.device) -> GPT:
    preset = model_config.get("preset")
    if preset:
        overrides = {k: v for k, v in model_config.items() if k != "preset"}
        model = GPT.from_preset(preset, **overrides)
    else:
        cfg = GPTConfig(**model_config)
        model = GPT(cfg)

    model = model.to(device)
    if is_main_process():
        param_count = model.count_parameters()
        print(f"Model: {param_count / 1e6:.1f}M parameters")
    return model


def main():
    parser = argparse.ArgumentParser(description="OrScale training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="Override config values: key.subkey=value")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    # Distributed setup
    rank, world_size, device = setup_distributed()
    if is_main_process():
        print(f"World size: {world_size}, Device: {device}")

    seed = config.get("training", {}).get("seed", 42)
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
                config=config,
            )
        except ImportError:
            print("wandb not installed, skipping W&B logging.")

    # Build model
    model = build_model(config.get("model", {}), device)

    # Build optimizer(s)
    opt_config = config.get("optimizer", {})
    opt_name = opt_config.get("name", "adamw")
    optimizers = build_optimizer(opt_name, model, opt_config)
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    # Build scheduler
    train_cfg = config.get("training", {})
    scheduler = CosineWithWarmup(
        optimizers,
        warmup_steps=train_cfg.get("warmup_steps", 500),
        max_steps=train_cfg.get("max_steps", 5000),
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.0),
    )

    # Build data loaders
    data_cfg = config.get("data", {})
    seq_len = config.get("model", {}).get("max_seq_len", 1024)
    batch_size = train_cfg.get("batch_size", 32)

    train_loader = create_dataloader(
        data_cfg, seq_len, batch_size, split="train",
        rank=rank, world_size=world_size, seed=seed,
    )

    val_loader = None
    val_pattern = data_cfg.get("val_pattern")
    if val_pattern:
        val_loader = create_dataloader(
            data_cfg, seq_len, batch_size, split="val",
            rank=rank, world_size=world_size, seed=seed,
        )

    # Build diagnostic logger
    diag_cfg = config.get("diagnostics", {})
    diag_logger = DiagnosticLogger(
        model=model,
        optimizers=optimizers,
        log_every=diag_cfg.get("log_every", 50),
        heavy_log_every=diag_cfg.get("heavy_log_every", 500),
        use_wandb=wandb_run is not None,
    )

    # Training config for Trainer
    trainer_config = {
        **train_cfg,
        "wandb_project": logging_cfg.get("wandb_project"),
    }

    # Build trainer and run
    trainer = Trainer(
        model=model,
        optimizers=optimizers,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        diagnostic_logger=diag_logger,
        device=device,
    )

    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
