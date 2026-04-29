#!/usr/bin/env python3
"""Analyze the CIFAR-10 / DavidNet optimizer sweep.

Parses every per-run log under one or more roots structured like

    sweeps/cifar10_sweeps/
        <timestamp>/
            name=<optimizer>_lr=<lr>_seed=<seed>.log
            sweep_config.json

produced by `scripts/sweep_cifar10.sh`. Each root may either be a parent of
timestamped sweep subdirs (canonical) or a single sweep dir whose ``.log``
files match the same naming pattern -- both layouts are merged transparently.
The default ``--sweeps-dir`` covers both ``sweeps/cifar10_sweeps`` (the
original 8-optimizer comparison) and ``sweeps/20260429_023319`` (the
follow-up muscale sweep). Emits:

    reports/cifar10_davidnet/
        runs.csv                 -- one row per run (aggregate metrics)
        epochs.csv               -- tidy per-epoch metrics for every run
        steps.csv                -- tidy per-step train metrics for every run
        summary_by_opt_lr.csv    -- (opt, lr) rollup, mean/std over seeds
        summary_by_opt_lr.md
        summary_best_lr.csv      -- best LR per optimizer + headline metric
        summary_best_lr.md
        plots/
            curves_val_top1.png
            curves_val_loss.png
            curves_train_loss.png
            lr_sensitivity_val_top1.png
            variance_best_lr_val_top1.png
            wallclock_seconds.png
        report.md                -- Markdown report tying everything together

The headline performance metric is the mean of the last 3 epochs' val_top1
(flag: --headline {last3,final,best}). Final-epoch and best-ever val_top1 are
also written to `runs.csv` for reference.

Usage:
    python scripts/analyze_cifar10_sweeps.py \
        --sweeps-dir sweeps/cifar10_sweeps \
        --out-dir reports/cifar10_davidnet

Dependencies: Python 3.8+, numpy, matplotlib. No pandas required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^name=(?P<opt>.+?)_lr=(?P<lr>[0-9eE+\-.]+)_seed=(?P<seed>\d+)\.log$"
)

STEP_RE = re.compile(
    r"^epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+\|\s+step\s+(?P<step>\d+)\s+\|\s+"
    r"loss\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+top1\s+(?P<top1>[-+0-9.eE]+)%\s+\|\s+"
    r"lr_mult\s+(?P<lr_mult>[-+0-9.eE]+)\s+\|\s+elapsed\s+(?P<elapsed>[-+0-9.eE]+)s"
)

EPOCH_END_RE = re.compile(
    r"^epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+END\s+\|\s+"
    r"val_loss\s+(?P<val_loss>[-+0-9.eE]+)\s+\|\s+"
    r"val_top1\s+(?P<val_top1>[-+0-9.eE]+)%\s+\|\s+"
    r"val_top5\s+(?P<val_top5>[-+0-9.eE]+)%"
)

DONE_RE = re.compile(
    r"^Training complete\.\s+(?P<epochs>\d+)\s+epochs in\s+(?P<seconds>[-+0-9.eE]+)s\."
)

SCHED_RE = re.compile(
    r"Scheduler:\s+\S+\s+\(steps_per_epoch=(?P<spe>\d+),\s+total_steps=(?P<total>\d+)\)"
)


@dataclass
class RunLog:
    optimizer: str
    lr: float
    seed: int
    sweep_dir: str
    log_path: str
    total_epochs: int = 0
    wallclock_s: float = math.nan
    steps_per_epoch: Optional[int] = None
    completed: bool = False
    epochs: List[Dict[str, float]] = field(default_factory=list)  # val metrics
    steps: List[Dict[str, float]] = field(default_factory=list)   # train metrics


def parse_filename(name: str) -> Optional[Tuple[str, float, int]]:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group("opt"), float(m.group("lr")), int(m.group("seed"))


def parse_log_file(path: Path) -> Optional[RunLog]:
    meta = parse_filename(path.name)
    if meta is None:
        return None
    opt, lr, seed = meta
    run = RunLog(
        optimizer=opt,
        lr=lr,
        seed=seed,
        sweep_dir=path.parent.name,
        log_path=str(path),
    )
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_END_RE.match(line)
            if m:
                run.epochs.append(
                    {
                        "epoch": int(m.group("epoch")),
                        "val_loss": float(m.group("val_loss")),
                        "val_top1": float(m.group("val_top1")),
                        "val_top5": float(m.group("val_top5")),
                    }
                )
                run.total_epochs = max(run.total_epochs, int(m.group("total_epochs")))
                continue
            m = STEP_RE.match(line)
            if m:
                run.steps.append(
                    {
                        "epoch": int(m.group("epoch")),
                        "step": int(m.group("step")),
                        "loss": float(m.group("loss")),
                        "top1": float(m.group("top1")),
                        "lr_mult": float(m.group("lr_mult")),
                        "elapsed_s": float(m.group("elapsed")),
                    }
                )
                continue
            m = DONE_RE.match(line)
            if m:
                run.completed = True
                run.wallclock_s = float(m.group("seconds"))
                run.total_epochs = max(run.total_epochs, int(m.group("epochs")))
                continue
            m = SCHED_RE.search(line)
            if m and run.steps_per_epoch is None:
                run.steps_per_epoch = int(m.group("spe"))
    return run


def collect_runs(sweeps_dir: Path) -> List[RunLog]:
    """Parse runs under ``sweeps_dir``.

    Two layouts are supported in the same call:

    1. ``sweeps_dir/<timestamp>/name=*_lr=*_seed=*.log`` -- the canonical
       layout produced by ``scripts/sweep_cifar10.sh`` (one timestamped
       subdir per (optimizer, lr, seed) sweep batch).
    2. ``sweeps_dir/name=*_lr=*_seed=*.log`` -- a single sweep batch passed
       in directly as the sweep dir.

    The recursive ``rglob`` matches both transparently.
    """
    runs: List[RunLog] = []
    for log in sorted(sweeps_dir.rglob("name=*_lr=*_seed=*.log")):
        r = parse_log_file(log)
        if r is not None:
            runs.append(r)
    return runs


def collect_runs_many(sweeps_dirs: Iterable[Path]) -> List[RunLog]:
    seen_paths: set[str] = set()
    runs: List[RunLog] = []
    for d in sweeps_dirs:
        for r in collect_runs(d):
            if r.log_path in seen_paths:
                continue
            seen_paths.add(r.log_path)
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def last_n_mean(xs: List[float], n: int) -> float:
    if not xs:
        return math.nan
    return float(np.mean(xs[-n:]))


def run_headline(run: RunLog, kind: str) -> float:
    xs = [e["val_top1"] for e in sorted(run.epochs, key=lambda e: e["epoch"])]
    if not xs:
        return math.nan
    if kind == "final":
        return xs[-1]
    if kind == "best":
        return max(xs)
    if kind == "last3":
        return last_n_mean(xs, 3)
    raise ValueError(f"unknown headline kind: {kind}")


def agg_by(runs: List[RunLog], key_fn) -> Dict[tuple, List[RunLog]]:
    g: Dict[tuple, List[RunLog]] = defaultdict(list)
    for r in runs:
        g[key_fn(r)].append(r)
    return g


def mean_std(values: Iterable[float]) -> Tuple[float, float, int]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, 0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), int(arr.size)


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown_table(path: Path, headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Stable color assignment for the 9 optimizers.
OPT_COLORS: Dict[str, str] = {
    "muon":                    "#1f77b4",
    "muon_moonlight":          "#17becf",
    "orscale_muon":            "#d62728",
    "orscale_muon_wd":         "#e377c2",
    "orscale_muon_moonlight":  "#ff7f0e",
    "mutrust":                 "#9467bd",
    "muscale":                 "#bcbd22",
    "adamw":                   "#2ca02c",
    "lamb":                    "#8c564b",
}
OPT_ORDER: List[str] = list(OPT_COLORS.keys())


def opt_color(opt: str) -> str:
    return OPT_COLORS.get(opt, "#555555")


def per_epoch_matrix(runs: List[RunLog], field_name: str, max_epochs: int) -> np.ndarray:
    """Return a (n_runs, max_epochs) array of `field_name`, padding with NaN."""
    mat = np.full((len(runs), max_epochs), np.nan, dtype=float)
    for i, run in enumerate(runs):
        for e in run.epochs:
            if 0 <= e["epoch"] < max_epochs:
                mat[i, e["epoch"]] = e[field_name]
    return mat


def plot_curves(
    runs_by_opt_best_lr: Dict[str, Tuple[float, List[RunLog]]],
    field_name: str,
    ylabel: str,
    out_path: Path,
    title: str,
    invert_better: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_epochs = max(
        (r.total_epochs for _, runs in runs_by_opt_best_lr.values() for r in runs),
        default=24,
    )
    for opt in OPT_ORDER:
        if opt not in runs_by_opt_best_lr:
            continue
        lr, runs = runs_by_opt_best_lr[opt]
        mat = per_epoch_matrix(runs, field_name, max_epochs)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(max_epochs)
        color = opt_color(opt)
        label = f"{opt} (lr={lr:g})"
        ax.plot(x, mean, color=color, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    if invert_better:
        ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_train_loss(
    runs_by_opt_best_lr: Dict[str, Tuple[float, List[RunLog]]],
    out_path: Path,
    smooth: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for opt in OPT_ORDER:
        if opt not in runs_by_opt_best_lr:
            continue
        lr, runs = runs_by_opt_best_lr[opt]
        # concatenate per-step (step, loss) for each run, then average by interp onto a common grid
        # simplest: average across seeds at each recorded step index
        lengths = [len(r.steps) for r in runs]
        if min(lengths) == 0:
            continue
        L = min(lengths)
        steps = np.asarray([r.steps[i]["step"] for r in runs for i in range(L)]).reshape(len(runs), L).mean(axis=0)
        losses = np.asarray([r.steps[i]["loss"] for r in runs for i in range(L)]).reshape(len(runs), L)
        mean = losses.mean(axis=0)
        std = losses.std(axis=0, ddof=1) if losses.shape[0] > 1 else np.zeros_like(mean)
        if smooth > 1 and mean.size > smooth:
            k = np.ones(smooth) / smooth
            mean = np.convolve(mean, k, mode="same")
            std = np.convolve(std, k, mode="same")
        color = opt_color(opt)
        ax.plot(steps, mean, color=color, label=f"{opt} (lr={lr:g})", linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Train step")
    ax.set_ylabel("Train loss (smoothed)")
    ax.set_title("Train loss — best LR per optimizer (mean ± std over seeds)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_lr_sensitivity(
    summary_opt_lr: List[dict],
    headline_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in summary_opt_lr:
        grouped[row["optimizer"]].append(row)
    for opt in OPT_ORDER:
        if opt not in grouped:
            continue
        rows = sorted(grouped[opt], key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rows]
        means = [r["headline_mean"] for r in rows]
        stds = [r["headline_std"] for r in rows]
        ax.errorbar(
            lrs,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=opt_color(opt),
            label=opt,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel(f"Val top-1 (%) — {headline_label}")
    ax.set_title("LR sensitivity — val_top1 vs learning rate")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_variance_best_lr(
    best_lr_runs: Dict[str, Tuple[float, List[RunLog]]],
    headline: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs: List[float] = []
    labels: List[str] = []
    all_vals: List[float] = []
    per_opt_vals: List[Tuple[int, List[float]]] = []
    for i, opt in enumerate([o for o in OPT_ORDER if o in best_lr_runs]):
        lr, runs = best_lr_runs[opt]
        vals = [run_headline(r, headline) for r in runs]
        per_opt_vals.append((i, vals))
        all_vals.extend(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.bar(
            i, mean, yerr=std,
            color=opt_color(opt), alpha=0.7, width=0.6,
            capsize=4, ecolor="#333333",
        )
        xs.append(i)
        labels.append(f"{opt}\nlr={lr:g}")

    for i, vals in per_opt_vals:
        ax.scatter(
            [i] * len(vals),
            vals,
            color="black",
            s=36,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # Zoom y-axis so seed-to-seed variation is actually visible.
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        pad = max(0.5, (hi - lo) * 0.25)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Val top-1 (%) — {headline}")
    ax.set_title("Per-seed val_top1 at best LR (bars = mean ± std, dots = individual seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_wallclock(runs: List[RunLog], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    by_opt: Dict[str, List[float]] = defaultdict(list)
    for r in runs:
        if not math.isnan(r.wallclock_s):
            by_opt[r.optimizer].append(r.wallclock_s)
    xs: List[float] = []
    labels: List[str] = []
    for i, opt in enumerate([o for o in OPT_ORDER if o in by_opt]):
        vals = by_opt[opt]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.bar(i, mean, yerr=std, color=opt_color(opt), alpha=0.75, capsize=4, width=0.6)
        xs.append(i)
        labels.append(opt)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock per run (s)")
    ax.set_title("Training wall-clock (mean ± std across all LRs and seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_runs_rows(runs: List[RunLog]) -> List[dict]:
    rows: List[dict] = []
    for r in runs:
        vtop1 = [e["val_top1"] for e in sorted(r.epochs, key=lambda e: e["epoch"])]
        vloss = [e["val_loss"] for e in sorted(r.epochs, key=lambda e: e["epoch"])]
        rows.append(
            {
                "optimizer": r.optimizer,
                "lr": r.lr,
                "seed": r.seed,
                "completed": int(r.completed),
                "num_epochs_observed": len(r.epochs),
                "total_epochs": r.total_epochs,
                "wallclock_s": f"{r.wallclock_s:.3f}" if not math.isnan(r.wallclock_s) else "",
                "final_val_top1": f"{vtop1[-1]:.4f}" if vtop1 else "",
                "best_val_top1": f"{max(vtop1):.4f}" if vtop1 else "",
                "last3_val_top1": f"{last_n_mean(vtop1, 3):.4f}" if vtop1 else "",
                "final_val_loss": f"{vloss[-1]:.4f}" if vloss else "",
                "best_val_loss": f"{min(vloss):.4f}" if vloss else "",
                "sweep_dir": r.sweep_dir,
                "log_path": r.log_path,
            }
        )
    return rows


def build_epochs_rows(runs: List[RunLog]) -> List[dict]:
    rows: List[dict] = []
    for r in runs:
        for e in r.epochs:
            rows.append(
                {
                    "optimizer": r.optimizer,
                    "lr": r.lr,
                    "seed": r.seed,
                    "epoch": e["epoch"],
                    "val_loss": f"{e['val_loss']:.6f}",
                    "val_top1": f"{e['val_top1']:.4f}",
                    "val_top5": f"{e['val_top5']:.4f}",
                }
            )
    return rows


def build_steps_rows(runs: List[RunLog]) -> List[dict]:
    rows: List[dict] = []
    for r in runs:
        for s in r.steps:
            rows.append(
                {
                    "optimizer": r.optimizer,
                    "lr": r.lr,
                    "seed": r.seed,
                    "epoch": s["epoch"],
                    "step": s["step"],
                    "train_loss": f"{s['loss']:.6f}",
                    "train_top1": f"{s['top1']:.4f}",
                    "lr_mult": f"{s['lr_mult']:.4f}",
                    "elapsed_s": f"{s['elapsed_s']:.3f}",
                }
            )
    return rows


def build_summary_by_opt_lr(runs: List[RunLog], headline: str) -> List[dict]:
    grouped = agg_by(runs, lambda r: (r.optimizer, r.lr))
    rows: List[dict] = []
    for (opt, lr), group in grouped.items():
        headline_vals = [run_headline(r, headline) for r in group]
        final_vals = [run_headline(r, "final") for r in group]
        best_vals = [run_headline(r, "best") for r in group]
        wct = [r.wallclock_s for r in group if not math.isnan(r.wallclock_s)]
        hmean, hstd, hn = mean_std(headline_vals)
        fmean, fstd, _ = mean_std(final_vals)
        bmean, bstd, _ = mean_std(best_vals)
        wmean, wstd, _ = mean_std(wct)
        rows.append(
            {
                "optimizer": opt,
                "lr": lr,
                "n_seeds": hn,
                "headline_mean": hmean,
                "headline_std": hstd,
                "final_mean": fmean,
                "final_std": fstd,
                "best_mean": bmean,
                "best_std": bstd,
                "wallclock_mean_s": wmean,
                "wallclock_std_s": wstd,
            }
        )
    rows.sort(key=lambda r: (OPT_ORDER.index(r["optimizer"]) if r["optimizer"] in OPT_ORDER else 99, r["lr"]))
    return rows


def pick_best_lr_per_opt(summary_opt_lr: List[dict]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for row in summary_opt_lr:
        opt = row["optimizer"]
        if math.isnan(row["headline_mean"]):
            continue
        cur = best.get(opt)
        if cur is None or row["headline_mean"] > cur["headline_mean"]:
            best[opt] = row
    return best


def fmt_pct(x: float) -> str:
    return "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"


def fmt_s(x: float) -> str:
    return "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.1f}"


def write_summary_markdown(summary: List[dict], out_path: Path, headline_label: str) -> None:
    headers = [
        "optimizer",
        "lr",
        "n_seeds",
        f"val_top1 {headline_label} (mean±std)",
        "final val_top1 (mean±std)",
        "best val_top1 (mean±std)",
        "wallclock s (mean±std)",
    ]
    rows = []
    for r in summary:
        rows.append(
            [
                r["optimizer"],
                f"{r['lr']:g}",
                str(r["n_seeds"]),
                f"{fmt_pct(r['headline_mean'])} ± {fmt_pct(r['headline_std'])}",
                f"{fmt_pct(r['final_mean'])} ± {fmt_pct(r['final_std'])}",
                f"{fmt_pct(r['best_mean'])} ± {fmt_pct(r['best_std'])}",
                f"{fmt_s(r['wallclock_mean_s'])} ± {fmt_s(r['wallclock_std_s'])}",
            ]
        )
    write_markdown_table(out_path, headers, rows, title=f"Summary by (optimizer, lr) — headline = {headline_label}")


def write_best_lr_markdown(best_lr: Dict[str, dict], out_path: Path, headline_label: str) -> None:
    headers = [
        "rank",
        "optimizer",
        "best lr",
        "n_seeds",
        f"val_top1 {headline_label} (mean±std)",
        "final val_top1 (mean±std)",
        "best val_top1 (mean±std)",
        "wallclock s (mean±std)",
    ]
    sorted_rows = sorted(best_lr.values(), key=lambda r: -r["headline_mean"])
    rows = []
    for i, r in enumerate(sorted_rows, 1):
        rows.append(
            [
                str(i),
                r["optimizer"],
                f"{r['lr']:g}",
                str(r["n_seeds"]),
                f"{fmt_pct(r['headline_mean'])} ± {fmt_pct(r['headline_std'])}",
                f"{fmt_pct(r['final_mean'])} ± {fmt_pct(r['final_std'])}",
                f"{fmt_pct(r['best_mean'])} ± {fmt_pct(r['best_std'])}",
                f"{fmt_s(r['wallclock_mean_s'])} ± {fmt_s(r['wallclock_std_s'])}",
            ]
        )
    write_markdown_table(out_path, headers, rows, title=f"Best LR per optimizer — ranked by {headline_label} val_top1")


def write_report(
    out_dir: Path,
    headline_label: str,
    runs: List[RunLog],
    summary_opt_lr: List[dict],
    best_lr: Dict[str, dict],
) -> None:
    """Emit ``report.md``.

    If a sibling file ``report_appendix.md`` exists in ``out_dir``, its
    contents are appended verbatim to the end of the auto-generated report.
    This lets a hand-written narrative (e.g. an optimizer-family deep dive)
    survive re-runs of the analysis pipeline without being overwritten.
    """
    lines: List[str] = []
    lines.append("# CIFAR-10 / DavidNet optimizer sweep — analysis report")
    lines.append("")
    lines.append(
        f"- Runs parsed: **{len(runs)}** "
        f"(completed: {sum(1 for r in runs if r.completed)})"
    )
    opts = sorted({r.optimizer for r in runs})
    lrs = sorted({r.lr for r in runs})
    seeds = sorted({r.seed for r in runs})
    lines.append(f"- Optimizers: {', '.join(opts)}")
    lines.append(f"- Learning rates swept: {', '.join(f'{lr:g}' for lr in lrs)}")
    lines.append(f"- Seeds: {', '.join(str(s) for s in seeds)}")
    lines.append(f"- Headline metric: **val_top1, {headline_label}** (mean over seeds, ± std dev)")
    lines.append("")

    lines.append("## Ranking at best LR")
    lines.append("")
    sorted_rows = sorted(best_lr.values(), key=lambda r: -r["headline_mean"])
    for i, r in enumerate(sorted_rows, 1):
        lines.append(
            f"{i}. **{r['optimizer']}** @ lr={r['lr']:g} — "
            f"{fmt_pct(r['headline_mean'])}% ± {fmt_pct(r['headline_std'])} "
            f"(final: {fmt_pct(r['final_mean'])}%, best-ever: {fmt_pct(r['best_mean'])}%)"
        )
    lines.append("")
    if sorted_rows:
        top = sorted_rows[0]
        lines.append(
            f"**Winner:** `{top['optimizer']}` at lr={top['lr']:g} with "
            f"{fmt_pct(top['headline_mean'])}% val_top1 ({headline_label}), "
            f"averaged over {top['n_seeds']} seeds."
        )
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("### Head-to-head at best LR (mean ± std over seeds)")
    lines.append("")
    lines.append("![val_top1](plots/curves_val_top1.png)")
    lines.append("")
    lines.append("![val_loss](plots/curves_val_loss.png)")
    lines.append("")
    lines.append("![train_loss](plots/curves_train_loss.png)")
    lines.append("")
    lines.append("### LR sensitivity")
    lines.append("")
    lines.append("![lr_sensitivity](plots/lr_sensitivity_val_top1.png)")
    lines.append("")
    lines.append("### Per-seed variance at best LR")
    lines.append("")
    lines.append("![variance](plots/variance_best_lr_val_top1.png)")
    lines.append("")
    lines.append("### Training wall-clock")
    lines.append("")
    lines.append("![wallclock](plots/wallclock_seconds.png)")
    lines.append("")

    lines.append("## Tables")
    lines.append("")
    lines.append("- Per (optimizer, lr) breakdown: `summary_by_opt_lr.md` / `.csv`")
    lines.append("- Best LR per optimizer: `summary_best_lr.md` / `.csv`")
    lines.append("- Raw tidy data: `runs.csv`, `epochs.csv`, `steps.csv`")
    lines.append("")

    appendix_path = out_dir / "report_appendix.md"
    if appendix_path.exists():
        appendix = appendix_path.read_text().rstrip()
        if appendix:
            lines.append(appendix)
            lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweeps-dir",
        type=Path,
        nargs="+",
        default=[
            repo_root / "sweeps" / "cifar10_sweeps",
            repo_root / "sweeps" / "20260429_023319",
        ],
        help=(
            "One or more roots to scan recursively for "
            "name=<opt>_lr=<lr>_seed=<seed>.log files. Each root may be "
            "either a parent of timestamped sweep subdirs (the canonical "
            "layout) or a single sweep dir; both layouts are merged."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "reports" / "cifar10_davidnet",
        help="Output directory for CSVs, plots, and the Markdown report.",
    )
    ap.add_argument(
        "--headline",
        choices=["last3", "final", "best"],
        default="last3",
        help="Headline val_top1 metric. 'last3' = mean of last 3 epochs (default).",
    )
    ap.add_argument(
        "--no-steps-csv",
        action="store_true",
        help="Skip writing steps.csv (it can get large).",
    )
    args = ap.parse_args()

    sweeps_dirs: List[Path] = (
        args.sweeps_dir if isinstance(args.sweeps_dir, list) else [args.sweeps_dir]
    )
    out_dir: Path = args.out_dir
    headline: str = args.headline
    headline_label = {
        "last3": "mean last-3 epochs",
        "final": "final epoch",
        "best": "best observed",
    }[headline]

    missing = [d for d in sweeps_dirs if not d.exists()]
    if missing:
        raise SystemExit(
            "sweeps dir(s) not found: " + ", ".join(str(d) for d in missing)
        )

    print(f"[info] Scanning sweeps: {', '.join(str(d) for d in sweeps_dirs)}")
    runs = collect_runs_many(sweeps_dirs)
    if not runs:
        raise SystemExit("No runs found. Check --sweeps-dir layout.")
    print(f"[info] Parsed {len(runs)} runs "
          f"({sum(1 for r in runs if r.completed)} completed).")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print("[info] Writing tidy CSVs...")
    write_csv(
        out_dir / "runs.csv",
        [
            "optimizer", "lr", "seed", "completed",
            "num_epochs_observed", "total_epochs", "wallclock_s",
            "final_val_top1", "best_val_top1", "last3_val_top1",
            "final_val_loss", "best_val_loss",
            "sweep_dir", "log_path",
        ],
        build_runs_rows(runs),
    )
    write_csv(
        out_dir / "epochs.csv",
        ["optimizer", "lr", "seed", "epoch", "val_loss", "val_top1", "val_top5"],
        build_epochs_rows(runs),
    )
    if not args.no_steps_csv:
        write_csv(
            out_dir / "steps.csv",
            ["optimizer", "lr", "seed", "epoch", "step",
             "train_loss", "train_top1", "lr_mult", "elapsed_s"],
            build_steps_rows(runs),
        )

    print("[info] Building summary tables...")
    summary_opt_lr = build_summary_by_opt_lr(runs, headline)
    write_csv(
        out_dir / "summary_by_opt_lr.csv",
        [
            "optimizer", "lr", "n_seeds",
            "headline_mean", "headline_std",
            "final_mean", "final_std",
            "best_mean", "best_std",
            "wallclock_mean_s", "wallclock_std_s",
        ],
        [
            {k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v)
                 else ("" if isinstance(v, float) and math.isnan(v) else v))
             for k, v in row.items()}
            for row in summary_opt_lr
        ],
    )
    write_summary_markdown(summary_opt_lr, out_dir / "summary_by_opt_lr.md", headline_label)

    best_lr_map = pick_best_lr_per_opt(summary_opt_lr)
    write_csv(
        out_dir / "summary_best_lr.csv",
        [
            "optimizer", "lr", "n_seeds",
            "headline_mean", "headline_std",
            "final_mean", "final_std",
            "best_mean", "best_std",
            "wallclock_mean_s", "wallclock_std_s",
        ],
        [
            {k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v)
                 else ("" if isinstance(v, float) and math.isnan(v) else v))
             for k, v in row.items()}
            for row in sorted(best_lr_map.values(), key=lambda r: -r["headline_mean"])
        ],
    )
    write_best_lr_markdown(best_lr_map, out_dir / "summary_best_lr.md", headline_label)

    print("[info] Building per-optimizer best-LR run groups...")
    best_lr_runs: Dict[str, Tuple[float, List[RunLog]]] = {}
    by_opt_lr: Dict[Tuple[str, float], List[RunLog]] = agg_by(runs, lambda r: (r.optimizer, r.lr))
    for opt, row in best_lr_map.items():
        runs_for_cell = by_opt_lr.get((opt, row["lr"]), [])
        best_lr_runs[opt] = (row["lr"], runs_for_cell)

    print("[info] Generating plots...")
    plot_curves(
        best_lr_runs,
        field_name="val_top1",
        ylabel="Val top-1 (%)",
        out_path=out_dir / "plots" / "curves_val_top1.png",
        title="Val top-1 — best LR per optimizer (mean ± std over seeds)",
    )
    plot_curves(
        best_lr_runs,
        field_name="val_loss",
        ylabel="Val loss",
        out_path=out_dir / "plots" / "curves_val_loss.png",
        title="Val loss — best LR per optimizer (mean ± std over seeds)",
    )
    plot_train_loss(best_lr_runs, out_dir / "plots" / "curves_train_loss.png")
    plot_lr_sensitivity(summary_opt_lr, headline_label, out_dir / "plots" / "lr_sensitivity_val_top1.png")
    plot_variance_best_lr(best_lr_runs, headline, out_dir / "plots" / "variance_best_lr_val_top1.png")
    plot_wallclock(runs, out_dir / "plots" / "wallclock_seconds.png")

    print("[info] Writing report.md...")
    write_report(out_dir, headline_label, runs, summary_opt_lr, best_lr_map)

    # Brief summary to stdout
    print()
    print("=" * 72)
    print(f" Ranking by val_top1 ({headline_label}) at best LR:")
    print("=" * 72)
    for i, row in enumerate(sorted(best_lr_map.values(), key=lambda r: -r["headline_mean"]), 1):
        print(
            f"  {i}. {row['optimizer']:<28s} lr={row['lr']:<6g}  "
            f"{row['headline_mean']:.2f}% ± {row['headline_std']:.2f}  "
            f"(n={row['n_seeds']}, wallclock≈{row['wallclock_mean_s']:.1f}s)"
        )
    print()
    print(f"[done] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
