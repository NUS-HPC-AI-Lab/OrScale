#!/usr/bin/env python3
"""
Moonlight-style scaling-law sweep driver.

For each ``(preset, tokens, optimizer)`` combination in the config, launch a
training run via ``scripts/train.py`` as a subprocess. Strict Moonlight-style
configs can specify the paper's 8K-context ``batch_examples``; this driver then
derives local micro-batch size, gradient accumulation, and training steps. After
all runs finish, parse their final validation loss from the per-run log, write a
CSV, fit ``L(C) = A * C^alpha`` per optimizer, and save ``scaling_law.png`` +
``scaling_law_fits.json``.

Usage:
    python scripts/run_scaling_law.py --config configs/scaling_law.yaml
    python scripts/run_scaling_law.py --config configs/scaling_law.yaml --dry-run

Config schema (see configs/scaling_law.yaml for a worked example):

    base_config: configs/small_125m.yaml         # loaded and overridden per run
    output_dir: results/scaling_law
    presets:
      - name: xs_400m
        params: 4.0e8
        tokens: 8.0e9
      - name: s_550m
        params: 5.5e8
        tokens: 1.1e10
      # ...
    optimizers:
      - name: adamw
        lr: 3.0e-4
      - name: muscale_alpha
        lr: 0.02
    seeds: 1
    launcher:
      command: python                            # e.g. torchrun
      extra_args: []                             # e.g. ["--nproc_per_node=8"]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orscale.analysis.scaling_law import (
    compute_pflop_s_days,
    fit_power_law,
    plot_pareto,
)


LOGGER = logging.getLogger("orscale.scaling_law")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FINAL_LOSS_RE = re.compile(r"[Ff]inal val_loss:\s*([0-9.]+)")
_LAST_VAL_LOSS_RE = re.compile(r"val_loss\s+([0-9.]+)")
_NPROC_RE = re.compile(r"--nproc_per_node(?:=|\s+)(\d+)")


def split_filter_values(values: list[str] | None) -> set[str] | None:
    """Parse repeated comma/space separated filter flags into a set."""
    if not values:
        return None

    selected: set[str] = set()
    for value in values:
        for part in value.replace(",", " ").split():
            if part:
                selected.add(part)
    return selected or None


def filter_by_name(items: list[dict], selected: set[str] | None, *, kind: str) -> list[dict]:
    """Filter config entries by their ``name`` while preserving config order."""
    if selected is None:
        return items

    known = {str(item["name"]) for item in items}
    unknown = selected - known
    if unknown:
        raise ValueError(
            f"Unknown {kind} filter value(s): {', '.join(sorted(unknown))}. "
            f"Known {kind}s: {', '.join(sorted(known))}"
        )
    return [item for item in items if str(item["name"]) in selected]


def infer_world_size(cfg: dict, launcher: dict) -> int:
    """Infer world size from config, falling back to launcher args when possible."""
    if "world_size" in cfg:
        return int(cfg["world_size"])

    extra_args = [str(arg) for arg in launcher.get("extra_args", [])]
    joined = " ".join(extra_args)
    match = _NPROC_RE.search(joined)
    if match:
        return int(match.group(1))
    return 1


def resolve_optimizer_lr(preset: dict, optimizer: dict) -> float | None:
    """Use optimizer-specific LR when present, otherwise the preset LR."""
    if "lr" in optimizer:
        return float(optimizer["lr"])
    if "lr" in preset:
        return float(preset["lr"])
    return None


def resolve_optimizer_value(value, *, lr: float | None):
    """Expand symbolic optimizer config values used by strict scaling configs."""
    if isinstance(value, str) and value == "same_as_lr":
        if lr is None:
            raise ValueError("optimizer value 'same_as_lr' requires an LR")
        return lr
    return value


def derive_training_overrides(preset: dict, cfg: dict, launcher: dict) -> tuple[list[str], dict]:
    """Derive per-preset batch/step overrides and metadata.

    Backward-compatible configs can continue to specify ``max_steps`` directly.
    Strict configs can specify:
      - ``seq_len``: context length.
      - ``batch_examples``: global batch measured in examples of ``seq_len``.
      - ``micro_batch_size``: local per-GPU batch size.
      - ``tokens``: target token budget.
    """
    overrides: list[str] = []
    metadata: dict[str, int | float | None] = {
        "seq_len": None,
        "micro_batch_size": None,
        "batch_examples": None,
        "grad_accum_steps": None,
        "tokens_per_step": None,
        "actual_tokens": None,
    }

    seq_len = preset.get("seq_len", preset.get("max_seq_len"))
    if seq_len is not None:
        seq_len = int(seq_len)
        overrides.append(f"model.max_seq_len={seq_len}")
        metadata["seq_len"] = seq_len

    batch_examples = preset.get("batch_examples")
    if batch_examples is None:
        max_steps = int(preset.get("max_steps", preset.get("steps", 2000)))
        overrides.append(f"training.max_steps={max_steps}")
        metadata["actual_tokens"] = float(preset["tokens"]) if "tokens" in preset else None
        return overrides, metadata

    if seq_len is None:
        raise ValueError(f"Preset {preset['name']} sets batch_examples but not seq_len")

    world_size = int(preset.get("world_size", infer_world_size(cfg, launcher)))
    micro_batch_size = int(preset.get("micro_batch_size", cfg.get("micro_batch_size", 1)))
    batch_examples = int(batch_examples)
    examples_per_micro_step = micro_batch_size * world_size
    if batch_examples % examples_per_micro_step != 0:
        raise ValueError(
            f"Preset {preset['name']} batch_examples={batch_examples} is not divisible by "
            f"micro_batch_size({micro_batch_size}) * world_size({world_size})"
        )

    grad_accum_steps = batch_examples // examples_per_micro_step
    if grad_accum_steps < 1:
        raise ValueError(f"Preset {preset['name']} derived grad_accum_steps < 1")

    tokens_per_step = batch_examples * seq_len
    tokens = float(preset["tokens"])
    max_steps = int(preset.get("max_steps", preset.get("steps", round(tokens / tokens_per_step))))
    if max_steps < 1:
        raise ValueError(f"Preset {preset['name']} derived max_steps < 1")

    actual_tokens = float(max_steps * tokens_per_step)
    overrides.extend([
        f"training.batch_size={micro_batch_size}",
        f"training.grad_accum_steps={grad_accum_steps}",
        f"training.max_steps={max_steps}",
    ])
    metadata.update({
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "batch_examples": batch_examples,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "actual_tokens": actual_tokens,
    })
    return overrides, metadata


def format_duration(seconds: float | None) -> str:
    """Format seconds as a compact human-readable duration."""
    if seconds is None:
        return "n/a"

    seconds = int(round(seconds))
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, _ = divmod(rem, 60)

    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def estimate_runtime_seconds(
    *,
    params: float,
    actual_tokens: float,
    cfg: dict,
    preset: dict,
    estimate_pflops_per_second: float | None = None,
) -> float | None:
    """Estimate wall time from token throughput or effective PFLOP/s."""
    if "estimate_tokens_per_second" in preset:
        tokens_per_second = float(preset["estimate_tokens_per_second"])
        if tokens_per_second <= 0:
            raise ValueError("estimate_tokens_per_second must be positive")
        return actual_tokens / tokens_per_second

    pflops_per_second = estimate_pflops_per_second
    if pflops_per_second is None:
        raw = preset.get("estimate_pflops_per_second", cfg.get("estimate_pflops_per_second"))
        pflops_per_second = float(raw) if raw is not None else None
    if pflops_per_second is None:
        return None
    if pflops_per_second <= 0:
        raise ValueError("estimate_pflops_per_second must be positive")

    flops = 6.0 * params * actual_tokens
    return flops / (pflops_per_second * 1e15)


def log_runtime_summary(rows: list[dict]) -> None:
    """Log one runtime estimate per preset."""
    seen: set[str] = set()
    for row in rows:
        preset_name = row["preset"]
        if preset_name in seen:
            continue
        seen.add(preset_name)
        duration = row.get("estimated_duration")
        if duration == "n/a":
            continue
        LOGGER.info(
            "Estimate per run: %-20s %s (%s steps, %.3g tokens)",
            preset_name,
            duration,
            row["max_steps"],
            row["actual_tokens"],
        )


def parse_final_val_loss(log_path: str) -> float | None:
    """Scan a training log for the last reported val_loss. Returns None if none found."""
    if not os.path.exists(log_path):
        return None
    best = None
    with open(log_path) as f:
        for line in f:
            m = _FINAL_LOSS_RE.search(line)
            if m:
                best = float(m.group(1))
        if best is not None:
            return best
        f.seek(0)
        for line in f:
            m = _LAST_VAL_LOSS_RE.search(line)
            if m:
                best = float(m.group(1))
    return best


def launch_run(
    base_config: str,
    overrides: list[str],
    log_path: str,
    launcher: dict,
) -> int:
    """Launch ``scripts/train.py`` with the given overrides and tee output to log_path."""
    cmd_parts: list[str] = [launcher.get("command", "python")]
    cmd_parts.extend(launcher.get("extra_args", []))
    cmd_parts.extend([str(Path(__file__).resolve().parent / "train.py"),
                      "--config", base_config, "--set", *overrides])
    LOGGER.info("Launching: %s", " ".join(cmd_parts))
    LOGGER.info("Writing log to %s", log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd_parts, stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="OrScale scaling-law sweep")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML describing the scaling-law sweep.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without launching.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only aggregate existing logs (for re-plotting).")
    parser.add_argument("--only-optimizer", action="append", default=None,
                        help="Run only selected optimizer name(s), comma- or space-separated.")
    parser.add_argument("--only-preset", action="append", default=None,
                        help="Run only selected preset name(s), comma- or space-separated.")
    parser.add_argument("--estimate-pflops-per-second", type=float, default=None,
                        help="Override effective PFLOP/s used for runtime estimates.")
    parser.add_argument("--set", nargs="*", default=[], dest="extra_overrides",
                        help="Additional key=value overrides appended to every training run.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_config = cfg["base_config"]
    output_dir = Path(cfg.get("output_dir", "results/scaling_law"))
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = cfg.get("launcher", {})
    presets = filter_by_name(
        cfg["presets"],
        split_filter_values(args.only_preset),
        kind="preset",
    )
    optimizers = filter_by_name(
        cfg["optimizers"],
        split_filter_values(args.only_optimizer),
        kind="optimizer",
    )
    seeds = int(cfg.get("seeds", 1))
    seed_base = int(cfg.get("seed_base", 42))

    csv_path = output_dir / "scaling_law.csv"
    rows: list[dict] = []

    # -- 1. Launch runs --
    for preset in presets:
        preset_name = preset["name"]
        model_preset = preset.get("model_preset", preset_name)
        params = float(preset["params"])
        tokens = float(preset["tokens"])
        training_overrides, training_meta = derive_training_overrides(preset, cfg, launcher)
        for optimizer in optimizers:
            opt_name = optimizer["name"]
            lr = resolve_optimizer_lr(preset, optimizer)
            for seed in range(seeds):
                run_seed = seed_base + seed
                run_id = f"{preset_name}-{opt_name}-seed{run_seed}"
                log_path = output_dir / f"log-{run_id}.log"

                overrides = [
                    f"model.preset={model_preset}",
                    *training_overrides,
                    f"training.seed={run_seed}",
                    f"optimizer.name={opt_name}",
                    f"logging.wandb_name={run_id}",
                ]
                if lr is not None:
                    overrides.append(f"optimizer.lr={lr}")
                for k, v in optimizer.items():
                    if k in {"name", "lr"}:
                        continue
                    resolved = resolve_optimizer_value(v, lr=lr)
                    overrides.append(f"optimizer.{k}={resolved}")
                for k, v in preset.get("overrides", {}).items():
                    overrides.append(f"{k}={v}")
                overrides.extend(args.extra_overrides)

                if args.dry_run:
                    LOGGER.info(
                        "[dry-run] %s -> %s  %s",
                        run_id,
                        log_path,
                        " ".join(overrides),
                    )
                    if training_meta["tokens_per_step"] is not None:
                        LOGGER.info(
                            "[dry-run] derived batch: examples=%s micro=%s accum=%s "
                            "tokens/step=%s actual_tokens=%.0f target_tokens=%.0f",
                            training_meta["batch_examples"],
                            training_meta["micro_batch_size"],
                            training_meta["grad_accum_steps"],
                            training_meta["tokens_per_step"],
                            training_meta["actual_tokens"],
                            tokens,
                        )
                elif not args.skip_training:
                    rc = launch_run(base_config, overrides, str(log_path), launcher)
                    if rc != 0:
                        LOGGER.error("Run %s exited with code %d", run_id, rc)

                final_loss = parse_final_val_loss(str(log_path))
                actual_tokens = float(training_meta["actual_tokens"] or tokens)
                pflop_days = compute_pflop_s_days(params, actual_tokens)
                estimated_seconds = estimate_runtime_seconds(
                    params=params,
                    actual_tokens=actual_tokens,
                    cfg=cfg,
                    preset=preset,
                    estimate_pflops_per_second=args.estimate_pflops_per_second,
                )
                estimated_duration = format_duration(estimated_seconds)
                row = {
                    "preset": preset_name,
                    "model_preset": model_preset,
                    "params": params,
                    "tokens": tokens,
                    "actual_tokens": actual_tokens,
                    "seq_len": training_meta["seq_len"],
                    "batch_examples": training_meta["batch_examples"],
                    "micro_batch_size": training_meta["micro_batch_size"],
                    "grad_accum_steps": training_meta["grad_accum_steps"],
                    "tokens_per_step": training_meta["tokens_per_step"],
                    "max_steps": next(
                        int(value.split("=", 1)[1])
                        for value in overrides
                        if value.startswith("training.max_steps=")
                    ),
                    "optimizer": opt_name,
                    "lr": lr,
                    "seed": run_seed,
                    "val_loss": final_loss if final_loss is not None else float("nan"),
                    "pflop_s_days": pflop_days,
                    "estimated_seconds": estimated_seconds if estimated_seconds is not None else "",
                    "estimated_duration": estimated_duration,
                    "log_path": str(log_path),
                }
                rows.append(row)

                if args.dry_run and estimated_seconds is not None:
                    LOGGER.info(
                        "[dry-run] estimated wall time: %s per run "
                        "(effective %.3g PFLOP/s)",
                        estimated_duration,
                        args.estimate_pflops_per_second
                        if args.estimate_pflops_per_second is not None
                        else preset.get(
                            "estimate_pflops_per_second",
                            cfg.get("estimate_pflops_per_second"),
                        ),
                    )

    # -- 2. Write CSV --
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %s (%d rows)", csv_path, len(rows))
        log_runtime_summary(rows)

    if args.dry_run:
        return

    # -- 3. Fit power laws per optimizer --
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if not (row["val_loss"] == row["val_loss"]):  # NaN check
            continue
        series.setdefault(row["optimizer"], []).append(
            (row["pflop_s_days"], row["val_loss"])
        )

    fits = {}
    fit_summary = {}
    for opt_name, pts in series.items():
        if len(pts) < 2:
            LOGGER.warning("Not enough points to fit power law for %s (have %d)",
                           opt_name, len(pts))
            continue
        xs, ys = zip(*pts)
        fit = fit_power_law(xs, ys, include_offset=False)
        fits[opt_name] = fit
        fit_summary[opt_name] = {"A": fit.A, "alpha": fit.alpha, "offset": fit.offset}
        LOGGER.info("%s: L(C) ≈ %.3g * C^%.4f", opt_name, fit.A, fit.alpha)

    fits_path = output_dir / "scaling_law_fits.json"
    with fits_path.open("w") as f:
        json.dump(fit_summary, f, indent=2)
    LOGGER.info("Wrote %s", fits_path)

    # -- 4. Plot --
    plot_path = output_dir / "scaling_law.png"
    plot_pareto(
        series,
        fits=fits,
        out_path=str(plot_path),
        title="Scaling law: validation loss vs compute",
    )
    LOGGER.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
