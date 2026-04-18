#!/usr/bin/env python3
"""
Run EleutherAI lm-evaluation-harness on an OrScale LM checkpoint.

Usage:
    python scripts/eval_downstream.py \
        --checkpoint checkpoints/pilot_25m/step_005000.pt \
        --tasks hellaswag,mmlu,gsm8k \
        --batch-size 8 \
        --limit 100

Set ``--wandb-project`` to log the results to a W&B run. Results are also
written to ``<checkpoint_dir>/eval_step_<step>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orscale.eval.downstream import run_downstream


LOGGER = logging.getLogger("orscale.eval_downstream")


def _flatten_results(results: dict) -> dict[str, float]:
    """Flatten the lm-eval results dict to a {task/metric: value} dict."""
    out: dict[str, float] = {}
    for task, metrics in results.get("results", {}).items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                out[f"{task}/{metric_name}"] = float(value)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="OrScale downstream LM evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a Trainer.save_checkpoint file.")
    parser.add_argument("--tasks", type=str, default="hellaswag",
                        help="Comma-separated lm-eval task names.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on test examples per task.")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: alongside the checkpoint).")
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    LOGGER.info("Loading checkpoint from %s", args.checkpoint)
    results = run_downstream(
        model_or_ckpt=args.checkpoint,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        device=device,
        num_fewshot=args.num_fewshot,
    )

    flat = _flatten_results(results)
    for key, value in sorted(flat.items()):
        LOGGER.info("%s = %.4f", key, value)

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else (
        ckpt_path.parent / f"eval_{ckpt_path.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"tasks": tasks, "metrics": flat, "raw": results.get("results", {})}, f, indent=2)
    LOGGER.info("Wrote %s", out_path)

    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=f"eval-{ckpt_path.stem}")
            wandb.log({f"downstream/{k}": v for k, v in flat.items()})
            wandb.finish()
        except ImportError:
            LOGGER.warning("wandb not installed, skipping W&B logging.")


if __name__ == "__main__":
    main()
