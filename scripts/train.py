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
import glob
import logging
import os
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orscale.model.gpt import GPT, GPTConfig, PRESET_CONFIGS
from orscale.model.moonlight_moe import MoonlightMoEForCausalLM
from orscale.optim import build_optimizer
from orscale.data.loader import create_dataloader
from orscale.diagnostics.logger import DiagnosticLogger
from orscale.training.trainer import Trainer
from orscale.training.scheduler import CosineWithWarmup
from orscale.utils.distributed import setup_distributed, cleanup_distributed, is_main_process


LOGGER = logging.getLogger("orscale.train")


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


def _wandb_slug(value) -> str:
    text = str(value).strip()
    return text.replace("/", "-").replace(" ", "-")


def build_wandb_run_metadata(config_path: str, config: dict) -> dict[str, object]:
    """Build readable W&B run metadata from the resolved config."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})
    logging_cfg = config.get("logging", {})

    config_name = Path(config_path).stem
    optimizer_name = _wandb_slug(opt_cfg.get("name", "adamw"))
    lr = opt_cfg.get("lr")
    seed = train_cfg.get("seed", 42)
    base_group = logging_cfg.get("wandb_group") or config_name
    explicit_name = logging_cfg.get("wandb_name")

    name_parts = [config_name, optimizer_name]
    if lr is not None:
        name_parts.append(f"lr{_wandb_slug(lr)}")
    name_parts.append(f"seed{_wandb_slug(seed)}")

    tags = [f"config:{config_name}", f"optimizer:{optimizer_name}", f"seed:{seed}"]
    if lr is not None:
        tags.append(f"lr:{lr}")
    preset = model_cfg.get("preset")
    if preset:
        tags.append(f"model:{preset}")

    return {
        "name": explicit_name or "-".join(name_parts),
        "group": base_group,
        "job_type": optimizer_name,
        "tags": tags,
    }


def create_tensorboard_writer(logging_cfg: dict, run_metadata: dict[str, object]):
    enabled = bool(
        logging_cfg.get("tensorboard", logging_cfg.get("tensorboard_enabled", False))
        or logging_cfg.get("tensorboard_log_dir")
    )
    if not enabled:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        log_main("tensorboard is not installed, skipping TensorBoard logging.")
        return None

    base_dir = logging_cfg.get("tensorboard_log_dir", "runs")
    group = _wandb_slug(run_metadata.get("group", "default"))
    name = _wandb_slug(run_metadata.get("name", "run"))
    log_dir = str(Path(base_dir) / group / name)
    writer = SummaryWriter(log_dir=log_dir)
    log_main("TensorBoard logging enabled: %s", log_dir)
    return writer


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


def _resolve_param_dtype(value) -> torch.dtype | None:
    if value is None:
        return None
    name = str(value).lower().strip()
    if name in {"none", "auto", "float32", "fp32"}:
        return torch.float32 if name in {"float32", "fp32"} else None
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported model.param_dtype: {value!r}")


def build_model(model_config: dict, device: torch.device) -> torch.nn.Module:
    architecture = str(model_config.get("architecture", "gpt")).lower().strip()
    param_dtype = _resolve_param_dtype(
        model_config.get("param_dtype", model_config.get("dtype"))
    )
    model_config = {
        k: v
        for k, v in model_config.items()
        if k not in {"architecture", "param_dtype", "dtype"}
    }

    if architecture in {"moonlight_moe", "moonlight-moe", "deepseek_moe", "deepseek-moe"}:
        preset = model_config.get("preset", "moonlight_16b_a3b")
        overrides = {k: v for k, v in model_config.items() if k != "preset"}
        model = MoonlightMoEForCausalLM.from_preset(preset, **overrides)
    elif architecture != "gpt":
        raise ValueError(
            f"Unknown model architecture: {architecture}. "
            "Choose 'gpt' or 'moonlight_moe'."
        )
    else:
        preset = model_config.get("preset")
        if preset:
            overrides = {k: v for k, v in model_config.items() if k != "preset"}
            model = GPT.from_preset(preset, **overrides)
        else:
            cfg = GPTConfig(**model_config)
            model = GPT(cfg)

    if param_dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=param_dtype)
    if is_main_process():
        param_count = model.count_parameters()
        LOGGER.info("Model: %.1fM parameters", param_count / 1e6)
        first_param = next(model.parameters(), None)
        if first_param is not None:
            LOGGER.info("Model parameter dtype: %s", first_param.dtype)
        if hasattr(model, "count_activated_parameters"):
            active_count = model.count_activated_parameters()
            LOGGER.info("Active parameters/token: %.1fM", active_count / 1e6)
    return model


def summarize_data_source(data_cfg: dict, split: str) -> str:
    pattern_key = "train_pattern" if split == "train" else "val_pattern"
    pattern = data_cfg.get(pattern_key)
    if pattern:
        matched_files = sorted(glob.glob(pattern))
        if matched_files:
            mode = "streaming" if data_cfg.get("streaming", False) else "in-memory"
            return f"{mode} .bin shards ({len(matched_files)} files) from {pattern}"
        return f".bin pattern configured but no files matched: {pattern}"

    hf_name = data_cfg.get("hf_dataset", "openwebtext")
    hf_split = "train" if split == "train" else "validation"
    max_tokens = data_cfg.get("max_tokens")
    max_tokens_str = f", max_tokens={max_tokens}" if max_tokens is not None else ""
    return f"HuggingFace dataset {hf_name!r} split={hf_split}{max_tokens_str}"


def log_run_summary(
    *,
    config_path: str,
    overrides: list[str],
    config: dict,
    rank: int,
    world_size: int,
    device: torch.device,
    optimizers: list[torch.optim.Optimizer],
    train_loader,
    val_loader,
    wandb_enabled: bool,
    tensorboard_enabled: bool,
) -> None:
    if not is_main_process():
        return

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    opt_cfg = config.get("optimizer", {})
    diag_cfg = config.get("diagnostics", {})
    logging_cfg = config.get("logging", {})

    seq_len = model_cfg.get("max_seq_len", 1024)
    batch_size = train_cfg.get("batch_size", 32)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    tokens_per_step = batch_size * seq_len * grad_accum_steps * world_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    log_main("Launching training run")
    log_main("  config: %s", config_path)
    log_main("  overrides: %s", overrides if overrides else "none")
    log_main(
        "  distributed: rank=%d local_rank=%d world_size=%d device=%s",
        rank, local_rank, world_size, device,
    )
    log_main(
        "  seed: base=%d effective_rank_seed=%d precision=%s",
        train_cfg.get("seed", 42),
        train_cfg.get("seed", 42) + rank,
        train_cfg.get("precision", "bfloat16"),
    )

    preset = model_cfg.get("preset", "custom")
    log_main(
        "  model: arch=%s preset=%s seq_len=%s norm=%s mlp=%s pos=%s tie_weights=%s",
        model_cfg.get("architecture", "gpt"),
        preset,
        seq_len,
        model_cfg.get("norm_type", model_cfg.get("rms_norm_eps", "default")),
        model_cfg.get("mlp_type", "moe" if model_cfg.get("architecture") else "default"),
        model_cfg.get("pos_encoding", "rope"),
        model_cfg.get("tie_weights", True),
    )
    log_main("  model dtype: %s", model_cfg.get("param_dtype", model_cfg.get("dtype", "float32")))
    log_main(
        "  optimizer: name=%s optimizers=%d zero_stage=%s lr=%s weight_decay=%s momentum=%s",
        opt_cfg.get("name", "adamw"),
        len(optimizers),
        opt_cfg.get("zero_stage", opt_cfg.get("zero", 0)),
        opt_cfg.get("lr", "default"),
        opt_cfg.get("weight_decay", "default"),
        opt_cfg.get("momentum", "n/a"),
    )
    if opt_cfg.get("name", "").lower() in {
        "muon",
        "muon_moonlight",
        "orscale",
        "orscale-lm",
        "orscale_lm",
        "orscale_original",
        "orscale_muon",
        "orscale_muon_wd",
        "orscale_muon_moonlight",
        "orscale_muon_moonlight_calibrated",
        "mutrust",
        "muscale",
        "muscale_alpha",
    }:
        log_main(
            "  optimizer extras: ns_iters=%s adamw_lr=%s alpha=%s r_min=%s r_max=%s c_denom=%s",
            opt_cfg.get("ns_iters", 5),
            opt_cfg.get("adamw_lr", "auto"),
            opt_cfg.get("alpha", "n/a"),
            opt_cfg.get("r_min", "n/a"),
            opt_cfg.get("r_max", "n/a"),
            opt_cfg.get("c_denom", "auto"),
        )

    log_main(
        "  schedule: max_steps=%d warmup_steps=%d grad_accum=%d tokens/step=%s",
        train_cfg.get("max_steps", 5000),
        train_cfg.get("warmup_steps", 500),
        grad_accum_steps,
        f"{tokens_per_step:,}",
    )
    log_main(
        "  training cadence: log_every=%s val_every=%s val_steps=%s save_every=%s",
        train_cfg.get("log_every", 10),
        train_cfg.get("val_every", 250),
        train_cfg.get("val_steps", 20),
        train_cfg.get("save_every", 0),
    )
    log_main(
        "  ddp: gradient_as_bucket_view=%s",
        train_cfg.get("ddp_gradient_as_bucket_view", True),
    )
    log_main(
        "  data(train): %s",
        summarize_data_source(data_cfg, "train"),
    )
    log_main(
        "  data(val): %s",
        summarize_data_source(data_cfg, "val") if val_loader is not None else "disabled",
    )
    log_main(
        "  loaders: train=%s val=%s",
        type(train_loader).__name__,
        type(val_loader).__name__ if val_loader is not None else "None",
    )
    log_main(
        "  diagnostics: log_every=%s heavy_log_every=%s",
        diag_cfg.get("log_every", 50),
        diag_cfg.get("heavy_log_every", 500),
    )
    if diag_cfg.get("selected_param_patterns"):
        log_main(
            "  diagnostics selected params: %s",
            ", ".join(diag_cfg.get("selected_param_patterns", [])),
        )
    log_main(
        "  wandb: %s%s",
        "enabled" if wandb_enabled else "disabled",
        f" (project={logging_cfg.get('wandb_project')}, group={logging_cfg.get('wandb_group')})"
        if wandb_enabled else "",
    )
    log_main(
        "  tensorboard: %s%s",
        "enabled" if tensorboard_enabled else "disabled",
        f" (dir={logging_cfg.get('tensorboard_log_dir', 'runs')})"
        if tensorboard_enabled else "",
    )


def main():
    setup_terminal_logging()

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
    log_main("World size: %d, Device: %s", world_size, device)

    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed + rank)

    run_metadata = build_wandb_run_metadata(args.config, config)

    # W&B init
    wandb_run = None
    logging_cfg = config.get("logging", {})
    if logging_cfg.get("wandb_project") and is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project=logging_cfg["wandb_project"],
                group=run_metadata["group"],
                name=run_metadata["name"],
                job_type=run_metadata["job_type"],
                tags=run_metadata["tags"],
                config=config,
            )
        except ImportError:
            log_main("wandb not installed, skipping W&B logging.")

    tensorboard_writer = (
        create_tensorboard_writer(logging_cfg, run_metadata)
        if is_main_process()
        else None
    )

    # Build model
    log_main("Building model...")
    model = build_model(config.get("model", {}), device)

    # Build optimizer(s)
    opt_config = config.get("optimizer", {})
    opt_name = opt_config.get("name", "adamw")
    log_main("Building optimizer(s): %s", opt_name)
    optimizers = build_optimizer(opt_name, model, opt_config)
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    # Build scheduler
    train_cfg = config.get("training", {})
    log_main("Building scheduler...")
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

    log_main("Building training data loader...")
    train_loader = create_dataloader(
        data_cfg, seq_len, batch_size, split="train",
        rank=rank, world_size=world_size, seed=seed,
    )

    val_loader = None
    val_pattern = data_cfg.get("val_pattern")
    if val_pattern:
        log_main("Building validation data loader...")
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
        tensorboard_writer=tensorboard_writer,
        log_per_param=diag_cfg.get("log_per_param", True),
        selected_param_patterns=diag_cfg.get("selected_param_patterns"),
        selected_metrics=diag_cfg.get("selected_metrics"),
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
        tensorboard_writer=tensorboard_writer,
    )

    log_run_summary(
        config_path=args.config,
        overrides=args.overrides,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device,
        optimizers=optimizers,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_enabled=wandb_run is not None,
        tensorboard_enabled=tensorboard_writer is not None,
    )
    log_main("Starting training loop...")
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()
    if tensorboard_writer is not None:
        tensorboard_writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
