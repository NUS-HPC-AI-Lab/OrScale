#!/usr/bin/env python3
"""
Moonlight-style scaling-law sweep driver.

For each ``(preset, tokens, optimizer)`` combination in the config, launch a
training run via ``scripts/train.py`` as a subprocess. After all runs finish,
parse their final validation loss from the per-run log, write a CSV, fit
``L(C) = A * C^alpha`` per optimizer via ``scipy.optimize.curve_fit``, and
save ``scaling_law.png`` + ``scaling_law_fits.json``.

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
import copy
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_config = cfg["base_config"]
    output_dir = Path(cfg.get("output_dir", "results/scaling_law"))
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = cfg["presets"]
    optimizers = cfg["optimizers"]
    seeds = int(cfg.get("seeds", 1))
    launcher = cfg.get("launcher", {})

    csv_path = output_dir / "scaling_law.csv"
    rows: list[dict] = []

    # -- 1. Launch runs --
    for preset in presets:
        preset_name = preset["name"]
        params = float(preset["params"])
        tokens = float(preset["tokens"])
        for optimizer in optimizers:
            opt_name = optimizer["name"]
            for seed in range(seeds):
                run_id = f"{preset_name}-{opt_name}-seed{42 + seed}"
                log_path = output_dir / f"log-{run_id}.log"

                overrides = [
                    f"model.preset={preset_name}",
                    f"training.max_steps={preset.get('max_steps', preset.get('steps', 2000))}",
                    f"training.seed={42 + seed}",
                    f"optimizer.name={opt_name}",
                ]
                for k, v in optimizer.items():
                    if k == "name":
                        continue
                    overrides.append(f"optimizer.{k}={v}")
                for k, v in preset.get("overrides", {}).items():
                    overrides.append(f"{k}={v}")

                if args.dry_run:
                    LOGGER.info("[dry-run] %s -> %s  %s",
                                run_id, log_path, " ".join(overrides))
                elif not args.skip_training:
                    rc = launch_run(base_config, overrides, str(log_path), launcher)
                    if rc != 0:
                        LOGGER.error("Run %s exited with code %d", run_id, rc)

                final_loss = parse_final_val_loss(str(log_path))
                pflop_days = compute_pflop_s_days(params, tokens)
                row = {
                    "preset": preset_name,
                    "params": params,
                    "tokens": tokens,
                    "optimizer": opt_name,
                    "seed": 42 + seed,
                    "val_loss": final_loss if final_loss is not None else float("nan"),
                    "pflop_s_days": pflop_days,
                    "log_path": str(log_path),
                }
                rows.append(row)

    # -- 2. Write CSV --
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %s (%d rows)", csv_path, len(rows))

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
