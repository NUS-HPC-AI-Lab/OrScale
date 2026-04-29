#!/usr/bin/env python3
"""Verify the 'dip -> bump -> dip' loss pattern on FineWeb-Edu small sweeps.

For three OrScale variants we suspect of showing a non-monotone loss curve
(``mutrust``, ``orscale_muon_moonlight``, ``muon_moonlight``), compare against
``muon`` and ``adamw`` to confirm only the former three families exhibit the
pattern.

Inputs:
    Per-run stdout logs written by ``scripts/sweep_fineweb_small.sh`` under
    ``sweeps/fineweb_<timestamp>/<cfg>-<opt>-lr<lr>-seed<seed>.log``.

Outputs (under ``reports/fineweb_bump/``):
    - ``summary.csv``      One row per run with bump detection metrics.
    - ``train_loss__<opt>.png`` / ``val_loss__<opt>.png``   Per-optimizer
       overlays of every LR, with dashed verticals at warmup end.
    - ``grid__train_loss.png`` / ``grid__val_loss.png``     Compact grid
       with all five optimizers side-by-side, same y-limits for apples-to-
       apples comparison.

Bump detection:
    A run is flagged as having the pattern when the training loss reaches a
    first local minimum (``loss_min1``), then climbs by at least
    ``--bump-threshold`` nats (default 0.1) to a local maximum
    (``loss_bump``), then falls again to a later minimum (``loss_min2``)
    with ``loss_min2 < loss_bump - bump_threshold``. We also report whether
    the validation-loss series has the same shape.

Usage:
    python scripts/analyze_fineweb_bump.py \
        --sweeps sweeps/fineweb_20260421_034858 sweeps/fineweb_20260421_035149 \
        --output reports/fineweb_bump
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

# Optimizer families we care about (flagged + controls).
# ``muscale`` was added on 2026-04-29 as a fourth flagged optimizer (Nesterov +
# RMS(M_hat) denominator + Moonlight shape norm) -- it has the same Moonlight-
# style shape rescaling that previously required widening the LR grid downward,
# so we expect it to need the same diagnostic treatment as the original three.
FLAGGED = ("mutrust", "orscale_muon_moonlight", "muon_moonlight", "muscale")
CONTROLS = ("muon", "adamw")
TARGET_OPTS = FLAGGED + CONTROLS

# Fallback warmup length (overridden by whatever we parse from the log).
DEFAULT_WARMUP = 1000

RUN_RX = re.compile(
    r"^(?P<cfg>[^/-]+)-(?P<opt>[a-z_]+)-lr(?P<lr>.+?)-seed(?P<seed>\d+)\.log$"
)
TRAIN_RX = re.compile(
    r"^step\s+(?P<step>\d+)/\d+\s+\|\s+loss\s+(?P<loss>[-\d.eE+nanif]+)"
)
VAL_RX = re.compile(
    r"^step\s+(?P<step>\d+)/\d+\s+\|\s+val_loss\s+(?P<loss>[-\d.eE+nanif]+)"
)
WARMUP_RX = re.compile(
    r"warmup_steps=(?P<warm>\d+)"
)


def parse_run_name(log_path: Path) -> dict | None:
    m = RUN_RX.match(log_path.name)
    if not m:
        return None
    fam = m.groupdict()
    # Normalize LR to a float for sorting/plotting.
    try:
        fam["lr_value"] = float(fam["lr"])
    except ValueError:
        fam["lr_value"] = math.nan
    return fam


def parse_log(log_path: Path) -> tuple[list, list, int]:
    """Return (train_series, val_series, warmup_steps).

    Each series is a list of (step:int, loss:float).
    """
    warmup = DEFAULT_WARMUP
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    with log_path.open("r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "warmup_steps=" in line:
                mm = WARMUP_RX.search(line)
                if mm:
                    warmup = int(mm.group("warm"))
            # Validation lines look like "step N/.. | val_loss V" and match
            # the train regex too; check val first.
            mv = VAL_RX.match(line)
            if mv:
                try:
                    val.append((int(mv.group("step")), float(mv.group("loss"))))
                except ValueError:
                    pass
                continue
            mt = TRAIN_RX.match(line)
            if mt:
                try:
                    train.append((int(mt.group("step")), float(mt.group("loss"))))
                except ValueError:
                    pass
    return train, val, warmup


def _smooth(
    series: list[tuple[int, float]], window: int,
) -> list[tuple[int, float]]:
    """Centered moving average in index space. Robust to a single bad log."""
    if window <= 1 or len(series) < window:
        return list(series)
    half = window // 2
    out = []
    for i in range(len(series)):
        lo = max(0, i - half)
        hi = min(len(series), i + half + 1)
        chunk = series[lo:hi]
        mean_v = sum(v for _, v in chunk) / len(chunk)
        out.append((series[i][0], mean_v))
    return out


def detect_bump(
    series: list[tuple[int, float]],
    *,
    warmup: int,
    threshold: float,
    search_end: int,
    skip_initial: int = 0,
    smooth_window: int = 5,
) -> dict:
    """Detect a down -> up -> down pattern.

    Strategy: scan only up to ``search_end`` (we're interested in whether the
    bump occurs around the warmup; later instabilities are a different story).
    Find the first local minimum. Then find the highest local max that occurs
    strictly after it. Then find the final minimum strictly after that max.
    The pattern is confirmed if ``min1 + threshold <= max`` and
    ``min2 + threshold <= max``.
    """
    if not series:
        return {"pattern": False, "min1": None, "bump": None, "min2": None}

    truncated = [(s, v) for s, v in series if s <= search_end]
    if skip_initial > 0 and len(truncated) > skip_initial:
        truncated = truncated[skip_initial:]
    truncated = _smooth(truncated, smooth_window)
    if len(truncated) < 5:
        return {"pattern": False, "min1": None, "bump": None, "min2": None}

    # min1: earliest global minimum up to a point (we want the first basin).
    # Use a running best until the loss has risen by threshold nats; that
    # locks in min1.
    min1_idx = 0
    min1_val = truncated[0][1]
    locked = False
    for i, (_s, v) in enumerate(truncated):
        if not locked:
            if v < min1_val:
                min1_val = v
                min1_idx = i
            elif v > min1_val + threshold:
                locked = True
        if locked:
            break

    if not locked:
        return {
            "pattern": False,
            "min1": (truncated[min1_idx][0], min1_val),
            "bump": None,
            "min2": None,
        }

    # bump: max after min1_idx.
    bump_idx = min1_idx + 1
    bump_val = -math.inf
    for i in range(min1_idx + 1, len(truncated)):
        _s, v = truncated[i]
        if v > bump_val:
            bump_val = v
            bump_idx = i

    # min2: min strictly after bump_idx (within truncated window).
    if bump_idx >= len(truncated) - 1:
        return {
            "pattern": False,
            "min1": (truncated[min1_idx][0], min1_val),
            "bump": (truncated[bump_idx][0], bump_val),
            "min2": None,
        }
    min2_idx = bump_idx + 1
    min2_val = truncated[bump_idx + 1][1]
    for i in range(bump_idx + 1, len(truncated)):
        _s, v = truncated[i]
        if v < min2_val:
            min2_val = v
            min2_idx = i

    # Require the bump to be *sustained*: count how many logged points
    # between min1 and min2 sit at least half a threshold above min1.
    # Single-batch loss spikes (occasionally seen with AdamW) have width 1
    # and are filtered out by this check.
    sustained_count = sum(
        1 for i in range(min1_idx + 1, min2_idx)
        if truncated[i][1] >= min1_val + threshold / 2
    )

    pattern = (
        min1_val + threshold <= bump_val
        and min2_val + threshold <= bump_val
        and sustained_count >= 3
    )
    return {
        "pattern": pattern,
        "min1": (truncated[min1_idx][0], min1_val),
        "bump": (truncated[bump_idx][0], bump_val),
        "min2": (truncated[min2_idx][0], min2_val),
        "sustained_count": sustained_count,
    }


def gather_runs(sweep_dirs: list[Path]) -> list[dict]:
    runs = []
    for d in sweep_dirs:
        if not d.is_dir():
            print(f"[warn] skipping non-dir: {d}")
            continue
        for log in sorted(d.glob("*.log")):
            meta = parse_run_name(log)
            if meta is None:
                continue
            if meta["opt"] not in TARGET_OPTS:
                continue
            train, val, warm = parse_log(log)
            meta.update(
                path=log,
                train=train,
                val=val,
                warmup=warm,
            )
            runs.append(meta)
    return runs


def cmap_for_lrs(lrs: list[float]):
    """Give each LR a distinct color, hotter = higher LR."""
    uniq = sorted(set(lrs))
    cmap = plt.get_cmap("viridis")
    return {lr: cmap(i / max(1, len(uniq) - 1)) for i, lr in enumerate(uniq)}


def plot_per_opt(runs: list[dict], out_dir: Path) -> None:
    by_opt: dict[str, list[dict]] = {}
    for r in runs:
        by_opt.setdefault(r["opt"], []).append(r)

    for opt, opt_runs in by_opt.items():
        lrs = sorted({r["lr_value"] for r in opt_runs})
        cmap = cmap_for_lrs(lrs)
        for series_key, ylabel in (("train", "Train loss"), ("val", "Val loss")):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            warmups = set()
            for r in sorted(opt_runs, key=lambda r: r["lr_value"]):
                data = r[series_key]
                if not data:
                    continue
                steps = [s for s, _ in data]
                losses = [v for _, v in data]
                ax.plot(
                    steps,
                    losses,
                    color=cmap[r["lr_value"]],
                    label=f"lr={r['lr']}",
                    linewidth=1.2,
                    alpha=0.9,
                )
                warmups.add(r["warmup"])
            for w in warmups:
                ax.axvline(w, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
            ax.set_title(f"{opt} ({ylabel}) — dashed = warmup end")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{series_key}_loss__{opt}.png", dpi=140)
            plt.close(fig)


def plot_grid(runs: list[dict], out_dir: Path) -> None:
    for series_key, ylabel in (("train", "Train loss"), ("val", "Val loss")):
        fig, axes = plt.subplots(
            1, len(TARGET_OPTS), figsize=(4 * len(TARGET_OPTS), 3.8), sharey=True
        )
        all_losses = []
        for opt, ax in zip(TARGET_OPTS, axes):
            opt_runs = [r for r in runs if r["opt"] == opt]
            lrs = sorted({r["lr_value"] for r in opt_runs})
            cmap = cmap_for_lrs(lrs)
            warmups = set()
            for r in sorted(opt_runs, key=lambda r: r["lr_value"]):
                data = r[series_key]
                if not data:
                    continue
                steps = [s for s, _ in data]
                losses = [v for _, v in data]
                all_losses.extend(losses)
                ax.plot(
                    steps, losses,
                    color=cmap[r["lr_value"]],
                    label=f"lr={r['lr']}",
                    linewidth=1.2, alpha=0.9,
                )
                warmups.add(r["warmup"])
            for w in warmups:
                ax.axvline(w, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
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
            f"{ylabel} across optimizers (FineWeb-Edu, small_125m)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"grid__{series_key}_loss.png", dpi=140)
        plt.close(fig)


def write_summary_csv(
    runs: list[dict], threshold: float, out_path: Path, skip_initial: int,
) -> None:
    fieldnames = [
        "opt", "lr", "seed", "cfg", "warmup",
        "train_pattern", "train_min1_step", "train_min1_loss",
        "train_bump_step", "train_bump_loss",
        "train_min2_step", "train_min2_loss",
        "train_delta_up", "train_delta_down",
        "val_pattern", "val_min1_step", "val_min1_loss",
        "val_bump_step", "val_bump_loss",
        "val_min2_step", "val_min2_loss",
        "log",
    ]
    rows = []
    for r in runs:
        search_end = max(
            (r["train"][-1][0] if r["train"] else 0),
            (r["val"][-1][0] if r["val"] else 0),
        )
        train_det = detect_bump(
            r["train"], warmup=r["warmup"], threshold=threshold,
            search_end=search_end, skip_initial=skip_initial,
        )
        val_det = detect_bump(
            r["val"], warmup=r["warmup"], threshold=threshold,
            search_end=search_end, skip_initial=0,
        )
        tmin1 = train_det["min1"]; tbump = train_det["bump"]; tmin2 = train_det["min2"]
        vmin1 = val_det["min1"]; vbump = val_det["bump"]; vmin2 = val_det["min2"]
        row = {
            "opt": r["opt"], "lr": r["lr"], "seed": r["seed"],
            "cfg": r["cfg"], "warmup": r["warmup"],
            "train_pattern": int(train_det["pattern"]),
            "train_min1_step": tmin1[0] if tmin1 else "",
            "train_min1_loss": f"{tmin1[1]:.4f}" if tmin1 else "",
            "train_bump_step": tbump[0] if tbump else "",
            "train_bump_loss": f"{tbump[1]:.4f}" if tbump else "",
            "train_min2_step": tmin2[0] if tmin2 else "",
            "train_min2_loss": f"{tmin2[1]:.4f}" if tmin2 else "",
            "train_delta_up":
                f"{tbump[1] - tmin1[1]:.4f}" if (tmin1 and tbump) else "",
            "train_delta_down":
                f"{tbump[1] - tmin2[1]:.4f}" if (tbump and tmin2) else "",
            "val_pattern": int(val_det["pattern"]),
            "val_min1_step": vmin1[0] if vmin1 else "",
            "val_min1_loss": f"{vmin1[1]:.4f}" if vmin1 else "",
            "val_bump_step": vbump[0] if vbump else "",
            "val_bump_loss": f"{vbump[1]:.4f}" if vbump else "",
            "val_min2_step": vmin2[0] if vmin2 else "",
            "val_min2_loss": f"{vmin2[1]:.4f}" if vmin2 else "",
            "log": str(r["path"]),
        }
        rows.append(row)

    rows.sort(key=lambda x: (x["opt"], x["lr"]))
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_group_summary(
    runs: list[dict], threshold: float, skip_initial: int,
) -> None:
    """Pretty print a pass/fail table grouped by optimizer."""
    print(
        f"\nBump detection (threshold = {threshold:.2f} nats, "
        "train-loss based)\n"
        + "-" * 72
    )
    by_opt: dict[str, list[dict]] = {}
    for r in runs:
        by_opt.setdefault(r["opt"], []).append(r)

    order = [o for o in TARGET_OPTS if o in by_opt]
    for opt in order:
        group = sorted(by_opt[opt], key=lambda r: r["lr_value"])
        flagged = opt in FLAGGED
        marker = "[FLAGGED]" if flagged else "[control]"
        print(f"  {opt:<26s} {marker}")
        for r in group:
            search_end = max(
                (r["train"][-1][0] if r["train"] else 0),
                (r["val"][-1][0] if r["val"] else 0),
            )
            det = detect_bump(
                r["train"], warmup=r["warmup"], threshold=threshold,
                search_end=search_end, skip_initial=skip_initial,
            )
            if det["pattern"]:
                tag = "dip->BUMP->dip"
                up = det["bump"][1] - det["min1"][1]
                dn = det["bump"][1] - det["min2"][1]
                extra = (
                    f"  bump@{det['bump'][0]} "
                    f"(+{up:.2f} / -{dn:.2f})"
                )
            else:
                tag = "monotone-ish"
                extra = ""
            print(f"      lr={r['lr']:<7s} seed={r['seed']:<3s}  {tag}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweeps", nargs="+", type=Path,
        default=[
            Path("sweeps/fineweb_20260421_034858"),
            Path("sweeps/fineweb_20260421_035149"),
        ],
        help="One or more sweep log directories.",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("reports/fineweb_bump"),
        help="Where to write CSV and PNGs.",
    )
    ap.add_argument(
        "--bump-threshold", type=float, default=0.3,
        help=(
            "Nats of rise AND later fall required to flag a bump. "
            "0.3 comfortably filters out warmup noise; the flagged runs "
            "show rises of 1-3 nats so this is not in danger of hiding "
            "genuine instability."
        ),
    )
    ap.add_argument(
        "--skip-initial", type=int, default=50,
        help="Ignore the first N training-loss points (pure warmup noise).",
    )
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    runs = gather_runs(args.sweeps)
    if not runs:
        print("[error] no matching runs found.")
        return 1

    print(f"Parsed {len(runs)} runs across {len(args.sweeps)} sweep dir(s).")

    write_summary_csv(
        runs, args.bump_threshold, args.output / "summary.csv", args.skip_initial,
    )
    plot_per_opt(runs, args.output)
    plot_grid(runs, args.output)
    print_group_summary(runs, args.bump_threshold, args.skip_initial)

    print(f"\nArtifacts written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
