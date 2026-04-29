#!/usr/bin/env python3
"""Decode local W&B run files for FineWeb small_125m trust-ratio diagnostics.

Reads ``wandb/run-*-<id>/run-<id>.wandb`` history records and summarizes
aggregate metrics logged every ``diagnostics.log_every`` steps (default 50).

Requires the Weights & Biases SDK (same version as training), e.g.::

    conda run -n orscale python scripts/analyze_fineweb_trust_ratio.py

Or with PYTHONPATH unset so ``import wandb`` resolves to the package, not
the repo's ``wandb/`` output directory — run from repo root with::

    cd /path/to/OrScale && PYTHONPATH= conda run -n orscale python scripts/analyze_fineweb_trust_ratio.py

Outputs CSV to stdout; use ``--markdown`` to append a table for reports.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _drop_repo_from_sys_path() -> None:
    """Avoid ``import wandb`` resolving to the repo's ``wandb/`` output folder."""
    root_res = _REPO_ROOT.resolve()
    cwd_res = Path.cwd().resolve()
    out: list[str] = []
    for p in sys.path:
        if p == "" or p == ".":
            if cwd_res == root_res:
                continue
            out.append(p)
            continue
        try:
            if Path(p).resolve() == root_res:
                continue
        except OSError:
            pass
        out.append(p)
    sys.path[:] = out


def parse_history(run_id: str, wandb_root: Path) -> list[dict]:
    _drop_repo_from_sys_path()
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2

    wdirs = list(wandb_root.glob(f"run-*-{run_id}"))
    if not wdirs:
        return []
    wandb_file = wdirs[0] / f"run-{run_id}.wandb"
    if not wandb_file.is_file():
        return []

    keys = (
        "diagnostics/_summary/clip_active_mean",
        "diagnostics/_summary/trust_ratio_raw_mean",
        "diagnostics/_summary/trust_ratio_raw_min",
        "diagnostics/_summary/trust_ratio_raw_max",
        "diagnostics/_summary/trust_ratio_clipped_mean",
        "diagnostics/_summary/trust_ratio_clipped_min",
        "diagnostics/_summary/trust_ratio_clipped_max",
    )
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    rows: list[dict] = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row: dict[str, float] = {}
        for it in rec.history.item:
            k = "/".join(it.nested_key) if it.nested_key else it.key
            if k not in keys and k not in ("_step", "step"):
                continue
            try:
                row[k] = float(json.loads(it.value_json))
            except Exception:
                try:
                    row[k] = float(it.value_json)
                except Exception:
                    pass
        if any(k in row for k in keys):
            rows.append(row)
    return rows


def stats(vals: list[float]) -> tuple[float | None, float | None, float | None]:
    if not vals:
        return None, None, None
    return min(vals), sum(vals) / len(vals), max(vals)


def main() -> int:
    _drop_repo_from_sys_path()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Print a markdown table after CSV header line",
    )
    args = ap.parse_args()
    root: Path = args.repo_root
    wandb_root = root / "wandb"

    opts = ("orscale_muon_moonlight", "mutrust", "muscale")
    log_dirs = [
        root / "sweeps/fineweb_20260427_014028",
        root / "sweeps/fineweb_20260427_061908",
        root / "sweeps/fineweb_20260429_024537",
        root / "sweeps/fineweb_20260429_120914",
    ]
    rx_run = re.compile(r"/runs/([A-Za-z0-9]+)")

    header = (
        "tag,opt,lr,run_id,n_diag,"
        "clip_active_eq1_frac,"
        "trust_clip_mean_min,trust_clip_mean_mean,trust_clip_mean_max,"
        "frac_clip_mean_eq_rmin,frac_clip_mean_eq_rmax,"
        "raw_mean_min,raw_mean_mean,raw_mean_max,val_loss"
    )
    print(header)

    md_rows: list[str] = []

    for d in log_dirs:
        if not d.is_dir():
            continue
        for log in sorted(d.glob("*.log")):
            opt = next((o for o in opts if f"-{o}-lr" in log.name), None)
            if not opt:
                continue
            lr = log.name.split(f"-{opt}-lr", 1)[1].split("-seed", 1)[0]
            text = log.read_text(errors="replace")
            ids = rx_run.findall(text)
            if not ids:
                continue
            run_id = ids[-1]
            if "20260429_024537" in str(log):
                tag = "muscale_bump"
            else:
                tag = "postfix"

            rows = parse_history(run_id, wandb_root)
            ca = [
                r["diagnostics/_summary/clip_active_mean"]
                for r in rows
                if "diagnostics/_summary/clip_active_mean" in r
            ]
            cm = [
                r["diagnostics/_summary/trust_ratio_clipped_mean"]
                for r in rows
                if "diagnostics/_summary/trust_ratio_clipped_mean" in r
            ]
            rm = [
                r["diagnostics/_summary/trust_ratio_raw_mean"]
                for r in rows
                if "diagnostics/_summary/trust_ratio_raw_mean" in r
            ]

            cand = list(wandb_root.glob(f"run-*-{run_id}/files/wandb-summary.json"))
            summary_path = cand[0] if cand else None
            val_loss = ""
            if summary_path and summary_path.is_file():
                try:
                    val_loss = str(json.loads(summary_path.read_text()).get("val/loss", ""))
                except Exception:
                    pass

            eq1 = sum(1 for x in ca if abs(x - 1.0) < 1e-12) / len(ca) if ca else None
            atmin = sum(1 for x in cm if abs(x - 0.5) < 1e-9) / len(cm) if cm else None
            atmax = sum(1 for x in cm if abs(x - 1.5) < 1e-9) / len(cm) if cm else None

            smin, smean, smax = stats(cm)
            rmin, rmean, rmax = stats(rm)

            print(
                ",".join(
                    str(x)
                    for x in (
                        tag,
                        opt,
                        lr,
                        run_id,
                        len(rows),
                        f"{eq1:.6f}" if eq1 is not None else "",
                        f"{smin:.6f}" if smin is not None else "",
                        f"{smean:.6f}" if smean is not None else "",
                        f"{smax:.6f}" if smax is not None else "",
                        f"{atmin:.6f}" if atmin is not None else "",
                        f"{atmax:.6f}" if atmax is not None else "",
                        f"{rmin:.6f}" if rmin is not None else "",
                        f"{rmean:.6f}" if rmean is not None else "",
                        f"{rmax:.6f}" if rmax is not None else "",
                        val_loss,
                    )
                )
            )
            if args.markdown:
                md_rows.append(
                    "| {tag} | `{opt}` | `{lr}` | `{rid}` | {n} | {eq1s} | {smeans} | {atmins} | {atmaxs} | {vl} |".format(
                        tag=tag,
                        opt=opt,
                        lr=lr,
                        rid=run_id,
                        n=len(rows),
                        eq1s=f"{eq1:.3f}" if eq1 is not None else "-",
                        smeans=f"{smean:.3f}" if smean is not None else "-",
                        atmins=f"{atmin:.3f}" if atmin is not None else "-",
                        atmaxs=f"{atmax:.3f}" if atmax is not None else "-",
                        vl=val_loss or "-",
                    )
                )

    if args.markdown and md_rows:
        print("\n## Table\n")
        print(
            "| tag | opt | lr | run_id | n_diag | frac_clip_eq1 | "
            "clip_mean_mean | frac@r_min | frac@r_max | val_loss |"
        )
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for line in md_rows:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
