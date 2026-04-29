#!/usr/bin/env python3
"""Compare post-fix FineWeb-Edu small_125m runs against the ``muon`` baseline.

This script is the successor to ``scripts/analyze_fineweb_bump.py``. The bump
script verified that ``mutrust``, ``muon_moonlight`` and ``orscale_muon_moonlight``
suffered a ``dip -> bump -> dip`` train-loss pattern on the original sweep
``sweeps/fineweb_20260421_034858`` / ``sweeps/fineweb_20260421_035149``.

After applying four fixes -- (1) widening the LR grid for the Moonlight-scaled
variants downward by ~10x, (2) tightening ``r_max``/``r_min`` from
``10.0/0.1`` to ``1.5/0.5``, (3) adding ``grad_clip_norm=1.0`` globally, and
(4) extra diagnostic aggregates -- this script answers four questions:

  Q1. Was the ``dip -> bump -> dip`` pattern eliminated?
  Q2. Which LR is each fixed optimizer's new optimum?
  Q3. How does the best post-fix run compare to ``muon`` at its best?
  Q4. Is the global grad-norm clip firing constantly (i.e. is the new effective
      LR set by the clip rather than by the schedule)?

A fourth flagged optimizer ``muscale`` was added on 2026-04-29. The latest
report uses the low-LR follow-up sweep ``sweeps/fineweb_20260429_120914``.
It uses the same post-fix defaults
(``r_min=0.5``, ``r_max=1.5``, ``grad_clip_norm=1.0``) and is treated as a
new (post-fix) optimizer with no pre-fix counterpart -- the before/after
panel is therefore skipped for it.

Inputs:
    --new-sweeps          : sweep dirs containing post-fix runs
                            (default: sweeps/fineweb_20260427_014028,
                                      sweeps/fineweb_20260427_061908,
                                      sweeps/fineweb_20260429_120914)
    --baseline-sweep      : dir containing the ``muon`` baseline runs
                            (default: sweeps/fineweb_20260421_034858)
    --old-flagged-sweeps  : pre-fix sweep dirs for the original three flagged
                            optimizers (no pre-fix sweep for muscale)
                            (default: sweeps/fineweb_20260421_034858,
                                      sweeps/fineweb_20260421_035149)

Outputs (under ``reports/fineweb_small/``):
    summary.csv                Per-run metrics (bump + final/best loss + grad-norm).
    summary_by_opt_lr.md       Markdown leaderboard.
    train_loss__<opt>.png      Per-optimizer LR overlay (post-fix).
    val_loss__<opt>.png        Same, validation.
    grid__train_loss.png       Side-by-side small multiples for the five optimizers.
    grid__val_loss.png         Same, validation.
    before_after__<opt>.png    Train-loss before/after the fix for each flagged opt.
    best_lr_comparison.png     Best LR per optimizer overlaid against muon@best.
    grad_norm__<opt>.png       Pre-clip grad-norm series.
    report.md                  Narrative summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_fineweb_bump import (
    parse_run_name,
    parse_log,
    detect_bump,
)


# Optimizers we want side-by-side in the post-fix comparison. ``muscale`` was
# added on 2026-04-29 as a fourth flagged optimizer once a dedicated sweep
# came in: it shares Moonlight's shape rescaling and exhibits the same dip->
# bump->dip pathology at the upper end of the original Muon LR grid (see
# ``reports/fineweb_bump_muscale/`` for the bump-detection summary on the new
# sweep). It does not have any *pre-fix* runs to compare against, so the
# before/after panel is skipped for it.
FLAGGED = ("muon_moonlight", "orscale_muon_moonlight", "mutrust", "muscale")
FLAGGED_WITH_BEFORE_AFTER = ("muon_moonlight", "orscale_muon_moonlight", "mutrust")
BASELINE = "muon"
TARGET_OPTS = (BASELINE,) + FLAGGED


GRAD_NORM_RX = re.compile(r"grad_norm\s+([\d.eE+-]+)")


def parse_grad_norm_series(log_path: Path) -> list[tuple[int, float]]:
    """Pull (step, grad_norm) from each `step ... | grad_norm V` line.

    Old (pre-fix) runs don't print grad_norm; the returned list is empty for
    those, which we treat as "clip disabled".
    """
    out: list[tuple[int, float]] = []
    step_rx = re.compile(r"^step\s+(\d+)/\d+")
    with log_path.open("r", errors="replace") as fh:
        for line in fh:
            mstep = step_rx.match(line)
            if not mstep or "val_loss" in line:
                continue
            mgn = GRAD_NORM_RX.search(line)
            if mgn:
                try:
                    out.append((int(mstep.group(1)), float(mgn.group(1))))
                except ValueError:
                    pass
    return out


def gather(
    sweep_dirs: list[Path],
    keep_opts: tuple[str, ...],
    *,
    tag: str,
) -> list[dict]:
    """Parse every log file matching ``keep_opts`` from ``sweep_dirs``.

    ``tag`` is a string like ``"new"`` or ``"old"`` carried on each run dict.
    """
    runs = []
    for d in sweep_dirs:
        if not d.is_dir():
            print(f"[warn] skip non-dir: {d}")
            continue
        for log in sorted(d.glob("*.log")):
            meta = parse_run_name(log)
            if meta is None or meta["opt"] not in keep_opts:
                continue
            train, val, warm = parse_log(log)
            grad = parse_grad_norm_series(log)
            meta.update(
                path=log, train=train, val=val, warmup=warm,
                grad=grad, tag=tag,
            )
            runs.append(meta)
    return runs


def final_window_mean(
    series: list[tuple[int, float]], n: int = 20,
) -> float | None:
    """Mean of the last ``n`` logged points (robust to any single-step jitter)."""
    if not series:
        return None
    n = min(n, len(series))
    return sum(v for _, v in series[-n:]) / n


def grad_clip_stats(
    grad: list[tuple[int, float]],
    *,
    warmup: int,
    threshold: float = 0.99,
) -> dict:
    """Stats on grad_norm (read off the *post-clip* `total_norm` returned by
    `torch.nn.utils.clip_grad_norm_`).

    With ``grad_clip_norm=1.0``, total_norm is bounded above by 1 plus a
    floating-point slack: any value >= ``threshold`` we treat as "clip
    saturated this step". We also report the post-warmup mean so the reader
    can see how aggressive the clip is *after* the LR has reached its peak.
    """
    if not grad:
        return {"n": 0, "mean": None, "saturated_frac": None,
                "post_warmup_mean": None, "post_warmup_sat_frac": None}
    vals = [v for _, v in grad]
    sat_frac = sum(1 for v in vals if v >= threshold) / len(vals)
    post = [v for s, v in grad if s > warmup]
    post_mean = sum(post) / len(post) if post else None
    post_sat = sum(1 for v in post if v >= threshold) / len(post) if post else None
    return {
        "n": len(grad),
        "mean": sum(vals) / len(vals),
        "saturated_frac": sat_frac,
        "post_warmup_mean": post_mean,
        "post_warmup_sat_frac": post_sat,
    }


def cmap_for_lrs(lrs: list[float]):
    uniq = sorted(set(lrs))
    cmap = plt.get_cmap("viridis")
    return {lr: cmap(i / max(1, len(uniq) - 1)) for i, lr in enumerate(uniq)}


def plot_per_opt(runs_new: list[dict], out_dir: Path) -> None:
    """One figure per optimizer overlaying every LR (post-fix)."""
    by_opt: dict[str, list[dict]] = {}
    for r in runs_new:
        by_opt.setdefault(r["opt"], []).append(r)
    for opt, group in by_opt.items():
        lrs = sorted({r["lr_value"] for r in group})
        cmap = cmap_for_lrs(lrs)
        for series_key, ylabel in (("train", "Train loss"), ("val", "Val loss")):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for r in sorted(group, key=lambda r: r["lr_value"]):
                data = r[series_key]
                if not data:
                    continue
                steps = [s for s, _ in data]
                losses = [v for _, v in data]
                ax.plot(steps, losses, color=cmap[r["lr_value"]],
                        label=f"lr={r['lr']}", linewidth=1.2, alpha=0.9)
            warm = group[0]["warmup"]
            ax.axvline(warm, color="k", linestyle="--", linewidth=0.7, alpha=0.4,
                       label="warmup end")
            ax.set_title(f"{opt} ({ylabel}) -- post-fix sweep")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{series_key}_loss__{opt}.png", dpi=140)
            plt.close(fig)


def plot_grid(runs_new: list[dict], out_dir: Path) -> None:
    """4-panel small-multiples (one panel per optimizer)."""
    for series_key, ylabel in (("train", "Train loss"), ("val", "Val loss")):
        fig, axes = plt.subplots(
            1, len(TARGET_OPTS), figsize=(4 * len(TARGET_OPTS), 3.8), sharey=True,
        )
        all_losses = []
        for opt, ax in zip(TARGET_OPTS, axes):
            opt_runs = [r for r in runs_new if r["opt"] == opt]
            lrs = sorted({r["lr_value"] for r in opt_runs})
            cmap = cmap_for_lrs(lrs)
            for r in sorted(opt_runs, key=lambda r: r["lr_value"]):
                data = r[series_key]
                if not data:
                    continue
                steps = [s for s, _ in data]
                losses = [v for _, v in data]
                all_losses.extend(losses)
                ax.plot(steps, losses, color=cmap[r["lr_value"]],
                        label=f"lr={r['lr']}", linewidth=1.2, alpha=0.9)
            warm = opt_runs[0]["warmup"] if opt_runs else 0
            ax.axvline(warm, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
            ax.set_title(opt)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")
        axes[0].set_ylabel(ylabel)
        if all_losses:
            lo = min(all_losses)
            hi = min(max(all_losses), lo + 6.0)
            for ax in axes:
                ax.set_ylim(lo - 0.2, hi + 0.2)
        fig.suptitle(
            f"{ylabel} (post-fix sweep, FineWeb-Edu small_125m)", fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"grid__{series_key}_loss.png", dpi=140)
        plt.close(fig)


def plot_before_after(
    runs_old: list[dict], runs_new: list[dict], out_dir: Path,
) -> None:
    """For each flagged opt: dashed = old (pre-fix), solid = new (post-fix).

    LRs differ between old and new for the Moonlight pair, so we don't try to
    color-match -- old is grey, new is viridis. The point of the plot is "no
    bump on any new curve, vs the old curves all bumping".

    Skipped for any flagged optimizer that does not have pre-fix runs (e.g.
    ``muscale``, which only has a post-fix sweep).
    """
    for opt in FLAGGED_WITH_BEFORE_AFTER:
        old = [r for r in runs_old if r["opt"] == opt]
        new = [r for r in runs_new if r["opt"] == opt]
        if not (old and new):
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in sorted(old, key=lambda r: r["lr_value"]):
            steps = [s for s, _ in r["train"]]
            losses = [v for _, v in r["train"]]
            ax.plot(steps, losses, color="grey", alpha=0.45, linewidth=0.9,
                    linestyle="--",
                    label=f"old lr={r['lr']}")
        lrs_new = sorted({r["lr_value"] for r in new})
        cmap = cmap_for_lrs(lrs_new)
        for r in sorted(new, key=lambda r: r["lr_value"]):
            steps = [s for s, _ in r["train"]]
            losses = [v for _, v in r["train"]]
            ax.plot(steps, losses, color=cmap[r["lr_value"]], linewidth=1.4,
                    alpha=0.95, label=f"new lr={r['lr']}")
        warm = new[0]["warmup"]
        ax.axvline(warm, color="k", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_title(f"{opt}: train loss before/after fix")
        ax.set_xlabel("step")
        ax.set_ylabel("Train loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"before_after__{opt}.png", dpi=140)
        plt.close(fig)


def best_run_per_opt(
    runs: list[dict], series_key: str = "val",
) -> dict[str, dict]:
    """Pick the run with the lowest final-window mean loss for each optimizer."""
    by_opt: dict[str, list[dict]] = {}
    for r in runs:
        by_opt.setdefault(r["opt"], []).append(r)
    out = {}
    for opt, group in by_opt.items():
        scored = []
        for r in group:
            m = final_window_mean(r[series_key], n=5)
            if m is None:
                continue
            scored.append((m, r))
        if scored:
            scored.sort(key=lambda t: t[0])
            out[opt] = scored[0][1]
    return out


def plot_best_lr_comparison(
    runs_new: list[dict], muon_runs: list[dict], out_dir: Path,
) -> None:
    """Overlay the best LR per optimizer on a single panel."""
    bests_new = best_run_per_opt(runs_new, series_key="val")
    bests_muon = best_run_per_opt(muon_runs, series_key="val")
    bests = {}
    if BASELINE in bests_muon:
        bests[BASELINE] = bests_muon[BASELINE]
    for k, v in bests_new.items():
        if k != BASELINE:
            bests[k] = v
    if not bests:
        return

    colors = {
        "muon": "black",
        "muon_moonlight": "tab:blue",
        "orscale_muon_moonlight": "tab:green",
        "mutrust": "tab:red",
        "muscale": "tab:orange",
    }
    for series_key, ylabel in (("train", "Train loss"), ("val", "Val loss")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for opt in TARGET_OPTS:
            r = bests.get(opt)
            if r is None:
                continue
            steps = [s for s, _ in r[series_key]]
            losses = [v for _, v in r[series_key]]
            ax.plot(steps, losses, color=colors.get(opt, None),
                    linewidth=1.4, alpha=0.95,
                    label=f"{opt} (lr={r['lr']})")
        ax.set_title(f"Best LR per optimizer -- {ylabel} (FineWeb-Edu small_125m)")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"best_lr_comparison__{series_key}.png", dpi=140)
        plt.close(fig)


def plot_grad_norm(runs_new: list[dict], out_dir: Path) -> None:
    by_opt: dict[str, list[dict]] = {}
    for r in runs_new:
        if r["grad"]:
            by_opt.setdefault(r["opt"], []).append(r)
    for opt, group in by_opt.items():
        lrs = sorted({r["lr_value"] for r in group})
        cmap = cmap_for_lrs(lrs)
        fig, ax = plt.subplots(figsize=(7, 4))
        for r in sorted(group, key=lambda r: r["lr_value"]):
            steps = [s for s, _ in r["grad"]]
            vals = [v for _, v in r["grad"]]
            ax.plot(steps, vals, color=cmap[r["lr_value"]],
                    linewidth=0.8, alpha=0.8, label=f"lr={r['lr']}")
        ax.axhline(1.0, color="r", linestyle="--", linewidth=0.7, alpha=0.5,
                   label="clip = 1.0")
        ax.set_title(f"{opt}: post-clip grad_norm vs step")
        ax.set_xlabel("step")
        ax.set_ylabel("grad_norm")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_norm__{opt}.png", dpi=140)
        plt.close(fig)


def write_summary(
    runs_all: list[dict], threshold: float, out_path: Path,
) -> list[dict]:
    """Write the per-run CSV; return the same rows for downstream use."""
    fields = [
        "tag", "opt", "lr", "seed", "warmup",
        "final_train_loss", "final_val_loss", "best_val_loss",
        "best_val_step",
        "train_pattern", "train_min1_loss", "train_bump_loss", "train_min2_loss",
        "train_delta_up", "train_delta_down",
        "grad_n", "grad_mean",
        "grad_post_warmup_mean", "grad_post_warmup_sat_frac",
        "log",
    ]
    rows = []
    for r in runs_all:
        search_end = max(
            (r["train"][-1][0] if r["train"] else 0),
            (r["val"][-1][0] if r["val"] else 0),
        )
        det = detect_bump(
            r["train"], warmup=r["warmup"], threshold=threshold,
            search_end=search_end, skip_initial=50,
        )
        gn = grad_clip_stats(r["grad"], warmup=r["warmup"])
        if r["val"]:
            best_step, best_val = min(r["val"], key=lambda t: t[1])
        else:
            best_step, best_val = "", ""
        row = {
            "tag": r["tag"], "opt": r["opt"], "lr": r["lr"], "seed": r["seed"],
            "warmup": r["warmup"],
            "final_train_loss": (
                f"{final_window_mean(r['train'], n=20):.4f}" if r["train"] else ""
            ),
            "final_val_loss": (
                f"{final_window_mean(r['val'], n=3):.4f}" if r["val"] else ""
            ),
            "best_val_loss": f"{best_val:.4f}" if r["val"] else "",
            "best_val_step": best_step,
            "train_pattern": int(det["pattern"]),
            "train_min1_loss": (
                f"{det['min1'][1]:.4f}" if det["min1"] else ""
            ),
            "train_bump_loss": (
                f"{det['bump'][1]:.4f}" if det["bump"] else ""
            ),
            "train_min2_loss": (
                f"{det['min2'][1]:.4f}" if det["min2"] else ""
            ),
            "train_delta_up": (
                f"{det['bump'][1] - det['min1'][1]:.4f}"
                if det["min1"] and det["bump"] else ""
            ),
            "train_delta_down": (
                f"{det['bump'][1] - det['min2'][1]:.4f}"
                if det["bump"] and det["min2"] else ""
            ),
            "grad_n": gn["n"],
            "grad_mean": (
                f"{gn['mean']:.3f}" if gn["mean"] is not None else ""
            ),
            "grad_post_warmup_mean": (
                f"{gn['post_warmup_mean']:.3f}"
                if gn["post_warmup_mean"] is not None else ""
            ),
            "grad_post_warmup_sat_frac": (
                f"{gn['post_warmup_sat_frac']:.3f}"
                if gn["post_warmup_sat_frac"] is not None else ""
            ),
            "log": str(r["path"]),
        }
        rows.append(row)
    rows.sort(key=lambda x: (x["tag"], x["opt"], float(x["lr"])))
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_leaderboard(rows: list[dict], out_md: Path) -> None:
    """Markdown leaderboard sorted by final val loss within each optimizer."""
    def lr_value(row: dict) -> float:
        try:
            return float(row["lr"])
        except (TypeError, ValueError):
            return math.inf

    lines = [
        "# FineWeb-Edu small_125m: post-fix sweep leaderboard",
        "",
        "Sorted by `final_val_loss` (mean of last 3 logged val checkpoints).",
        "",
        "| tag | optimizer | lr | final_train | final_val | best_val |"
        " bump | grad_post_warmup_mean | grad_post_warmup_sat_frac |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda x: (x["opt"], x["tag"], lr_value(x))):
        lines.append(
            f"| {r['tag']} | {r['opt']} | {r['lr']} | "
            f"{r['final_train_loss']} | {r['final_val_loss']} | "
            f"{r['best_val_loss']} | "
            f"{'YES' if r['train_pattern'] == 1 else 'no'} | "
            f"{r['grad_post_warmup_mean'] or '-'} | "
            f"{r['grad_post_warmup_sat_frac'] or '-'} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--new-sweeps", nargs="+", type=Path,
        default=[
            Path("sweeps/fineweb_20260427_014028"),
            Path("sweeps/fineweb_20260427_061908"),
            # muscale-only follow-up sweep (2026-04-29). Same fix bundle is
            # in effect (r_min=0.5, r_max=1.5, grad_clip=1.0) so this is
            # treated as a "new" / post-fix sweep.
            Path("sweeps/fineweb_20260429_120914"),
        ],
    )
    ap.add_argument(
        "--baseline-sweep", type=Path,
        default=Path("sweeps/fineweb_20260421_034858"),
    )
    ap.add_argument(
        "--old-flagged-sweeps", nargs="+", type=Path,
        default=[
            Path("sweeps/fineweb_20260421_034858"),
            Path("sweeps/fineweb_20260421_035149"),
        ],
    )
    ap.add_argument(
        "--output", type=Path, default=Path("reports/fineweb_small"),
    )
    ap.add_argument("--bump-threshold", type=float, default=0.3)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Post-fix runs for the three flagged optimizers.
    runs_new = gather(args.new_sweeps, FLAGGED, tag="new")
    # 2. Baseline muon (only present in the old sweep dir).
    runs_muon = gather([args.baseline_sweep], (BASELINE,), tag="baseline")
    # 3. Pre-fix runs of the three flagged optimizers (for before/after).
    runs_old = gather(args.old_flagged_sweeps, FLAGGED, tag="old")

    print(
        f"Parsed {len(runs_new)} new flagged runs, "
        f"{len(runs_muon)} muon baseline runs, "
        f"{len(runs_old)} old flagged runs."
    )

    # The "post-fix-vs-baseline" view that headlines the report.
    post_fix_view = runs_new + runs_muon

    plot_per_opt(post_fix_view, args.output)
    plot_grid(post_fix_view, args.output)
    plot_before_after(runs_old, runs_new, args.output)
    plot_best_lr_comparison(runs_new, runs_muon, args.output)
    plot_grad_norm(runs_new, args.output)

    rows = write_summary(
        runs_new + runs_muon + runs_old,
        args.bump_threshold,
        args.output / "summary.csv",
    )
    write_leaderboard(rows, args.output / "summary_by_opt_lr.md")

    print(f"\nArtifacts written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
