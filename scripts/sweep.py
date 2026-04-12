#!/usr/bin/env python3
"""
Hyperparameter sweep launcher for OrScale experiments.

Generates and launches training runs for all combinations of sweep parameters.
Runs can be executed sequentially or in parallel via subprocess.

Usage:
    python scripts/sweep.py --config configs/pilot_25m.yaml \
        --sweep optimizer.name=muon,muscale_alpha optimizer.lr=0.01,0.02,0.05 \
        --seeds 3

    python scripts/sweep.py --config configs/pilot_25m.yaml \
        --sweep optimizer.name=adamw,muon,muon_moonlight,orscale_muon,orscale_muon_wd,orscale_muon_moonlight,mutrust,muscale,muscale_alpha \
        --seeds 1 --parallel 4
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime


def parse_sweep_spec(specs: list[str]) -> dict[str, list[str]]:
    """Parse sweep specs like 'key=val1,val2,val3' into a dict."""
    result = {}
    for spec in specs:
        key, _, values = spec.partition("=")
        if not values:
            raise ValueError(f"Bad sweep spec: {spec}. Expected key=val1,val2,...")
        result[key] = values.split(",")
    return result


def generate_runs(sweep_params: dict[str, list[str]], num_seeds: int) -> list[dict[str, str]]:
    """Generate all combinations of sweep params x seeds."""
    keys = list(sweep_params.keys())
    value_lists = list(sweep_params.values())
    runs = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(keys, combo))
        for seed in range(num_seeds):
            run = copy.copy(overrides)
            run["training.seed"] = str(42 + seed)
            runs.append(run)
    return runs


def run_name(overrides: dict[str, str]) -> str:
    """Generate a short name for a run from its overrides."""
    parts = []
    for k, v in overrides.items():
        short_key = k.split(".")[-1]
        parts.append(f"{short_key}={v}")
    return "_".join(parts)


_PRINT_LOCK = threading.Lock()


def emit(message: str = "", *, end: str = "\n") -> None:
    """Print a message without interleaving across streaming threads."""
    with _PRINT_LOCK:
        print(message, end=end, flush=True)


def stream_output(
    proc: subprocess.Popen[str],
    log_path: str,
    name: str,
    *,
    stream: bool,
) -> None:
    """Write subprocess output to a log file and optionally mirror it live."""
    if proc.stdout is None:
        raise RuntimeError("Subprocess stdout is not available for streaming.")

    with open(log_path, "w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if stream:
                emit(f"[{name}] {line}", end="")

    proc.stdout.close()


def launch_run(
    config_path: str,
    overrides: dict[str, str],
    gpu_id: int | None = None,
) -> subprocess.Popen[str]:
    """Launch a single training run as a subprocess."""
    cmd = [sys.executable, "scripts/train.py", "--config", config_path]
    override_args = [f"{k}={v}" for k, v in overrides.items()]
    if override_args:
        cmd += ["--set"] + override_args

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    name = run_name(overrides)
    emit(f"Launching: {name}")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def main():
    parser = argparse.ArgumentParser(description="OrScale HP sweep")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML")
    parser.add_argument("--sweep", nargs="+", required=True,
                        help="Sweep specs: key=val1,val2,...")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds per combo")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max parallel runs (default: sequential)")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable live streaming of train.py logs")
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    sweep_params = parse_sweep_spec(args.sweep)
    runs = generate_runs(sweep_params, args.seeds)

    emit(f"Sweep: {len(runs)} total runs")
    emit(f"  Params: {json.dumps(sweep_params, indent=2)}")
    emit(f"  Seeds: {args.seeds}")
    emit(f"  Live logs: {'enabled' if args.stream else 'disabled'}")
    emit()

    if args.dry_run:
        for i, run in enumerate(runs):
            emit(f"  [{i+1}] {run_name(run)}")
        return

    # Create sweep log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = f"sweeps/{timestamp}"
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "sweep_config.json"), "w") as f:
        json.dump({"config": args.config, "sweep": sweep_params, "seeds": args.seeds, "runs": runs}, f, indent=2)

    if args.parallel <= 1:
        for i, run in enumerate(runs):
            name = run_name(run)
            emit(f"\n--- Run {i+1}/{len(runs)}: {name} ---")
            proc = launch_run(args.config, run)

            log_path = os.path.join(sweep_dir, f"{name}.log")
            stream_thread = threading.Thread(
                target=stream_output,
                args=(proc, log_path, name),
                kwargs={"stream": args.stream},
                daemon=True,
            )
            stream_thread.start()
            proc.wait()
            stream_thread.join()

            if proc.returncode != 0:
                emit(f"  FAILED (exit code {proc.returncode})")
            else:
                emit("  Done.")
    else:
        # Parallel execution with limited concurrency
        active: list[dict[str, object]] = []
        run_queue = list(runs)

        while run_queue or active:
            # Launch new runs if slots available
            while run_queue and len(active) < args.parallel:
                run = run_queue.pop(0)
                gpu_id = len(active) % max(1, args.parallel)
                name = run_name(run)
                proc = launch_run(args.config, run, gpu_id=gpu_id)
                log_path = os.path.join(sweep_dir, f"{name}.log")
                stream_thread = threading.Thread(
                    target=stream_output,
                    args=(proc, log_path, name),
                    kwargs={"stream": args.stream},
                    daemon=True,
                )
                stream_thread.start()
                active.append({
                    "proc": proc,
                    "name": name,
                    "thread": stream_thread,
                })

            still_active = []
            for entry in active:
                proc = entry["proc"]
                name = entry["name"]
                stream_thread = entry["thread"]
                ret = proc.poll()
                if ret is not None and not stream_thread.is_alive():
                    stream_thread.join()
                    status = "OK" if ret == 0 else f"FAIL({ret})"
                    emit(f"  Finished: {name} [{status}]")
                else:
                    still_active.append(entry)
            active = still_active

            if active:
                time.sleep(0.5)

    emit(f"\nSweep complete. Logs in {sweep_dir}/")


if __name__ == "__main__":
    main()
