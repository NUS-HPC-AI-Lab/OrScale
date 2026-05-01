#!/usr/bin/env bash
# Sweep all 8 OrScale memo variants + AdamW + LAMB on CIFAR-10 / DavidNet.
#
# Hardware target  : 2 x A100-40G (single node, no DDP).
#   Each run uses 1 GPU, batch_size=512, matching the Muon blog reference.
#   sweep.py launches `--parallel 2` runs at a time, assigning CUDA_VISIBLE_DEVICES
#   per run (logical device 0 inside the child process maps to one physical GPU).
#
# Optimizers       : 8 Muon-family + AdamW + LAMB
# Per-family LR    : Muon family {0.005, 0.01, 0.02, 0.04} for seven variants
#                    (muon, muon_moonlight, orscale_muon, orscale_muon_wd,
#                    orscale_muon_moonlight, orscale_muon_moonlight_calibrated,
#                    mutrust)
#                    muscale       {0.04, 0.06, 0.08, 0.12} (follow-up grid;
#                    override with MUSCALE_LRS=...)
#                    AdamW/LAMB    {1e-3, 3e-3, 1e-2}
# Per-variant trust-ratio clip:
#                    Default (YAML)                       r_min=0.5, r_max=1.5
#                    orscale_muon_moonlight             → r_min=0.1, r_max=5.0
#                    orscale_muon_moonlight_calibrated  → r_min=0.1, r_max=5.0
#                    (Moonlight-shape variants get LARS/LAMB-style looser
#                    bounds: the calibrated denominator is auto-set so
#                    r̂(0)=1 per layer; the original Moonlight variant's
#                    natural r̂ range is also wider than the tight default)
# Seeds per cell   : 3
# Total runs       : (8 * 4 + 2 * 3) * 3 = 114
#
# The 7th Muon-family variant is `muscale` (added 2026-04-28): mutrust's
# trust ratio (||W||_F / ||M_hat||_F, identical to OrScale2's after
# canceling the sqrt(mn) factor) plus Moonlight's static shape factor
# 0.2*sqrt(max(m,n)) on the orthogonalized update Q. It is the
# (dynamic-denominator, shape-scaled) corner of the 2x2 design space
# spanned by trust-ratio denominator x shape factor, and is the
# recommended primary OrScale variant in the paper.
#
# Why the Moonlight-scaled variants (muon_moonlight, orscale_muon_moonlight)
# stay on the Muon grid here, unlike sweep_fineweb_small.sh
# (update 2026-04-22): the `0.2 * sqrt(max(m, n))` shape constant inflates
# the per-entry update by ~5-14x on DavidNet's largest flattened conv
# (512 x 4608). On FineWeb that same inflation, combined with 20k training
# steps on a pre-norm transformer, produced a sustained dip -> bump -> dip
# loss pattern (see reports/fineweb_bump/). On CIFAR the same optimizers
# are *stable* at every LR in the Muon grid -- three stabilizing factors
# do not apply to LM training:
#   1. Training is ~10x shorter (~2350 vs. 20000 steps).
#   2. DavidNet has BatchNorm, which re-centers activations every step.
#   3. reports/cifar10_davidnet/summary_by_opt_lr.md shows each Muon-family
#      optimizer peaks *inside* {0.005..0.04} -- no variant wants AdamW-
#      scale LRs here. Moving them to the AdamW grid would under-sample the
#      empirical optimum.
# muscale shares the same shape factor as muon_moonlight and
# orscale_muon_moonlight, so the same three stabilizers apply and the
# Muon grid was the right *first* sweep. The 2026-04-29 muscale-only
# results (reports/cifar10_davidnet/report_appendix.md) show the best
# cell at lr=0.04 on the boundary of {0.005..0.04} with a flat surface
# (93.56-93.75% across the grid). The follow-up grid widens **upward**
# only: {0.04, 0.06, 0.08, 0.12}. Override at launch time with
# `MUSCALE_LRS=0.04,0.05,0.06` if you want a custom bracket.
#
# Usage:
#   bash scripts/sweep_cifar10.sh                 # full sweep
#   DRY_RUN=1 bash scripts/sweep_cifar10.sh       # print runs without launching
#   PARALLEL=1 bash scripts/sweep_cifar10.sh      # force sequential (e.g. 1 GPU)
#   OPTIMIZERS="muscale" \
#     bash scripts/sweep_cifar10.sh               # optimizer subset (comma- or
#                                                 # space-separated)
#
# Notes:
#   - Run from the repo root.
#   - Make sure CIFAR-10 has been downloaded:
#       python scripts/prepare_vision_data.py --dataset cifar10
#   - W&B run names are auto-set to `<wandb_group>-<opt>-lr<lr>-seed<seed>`
#     so each cell shows up distinctly in the dashboard.

set -euo pipefail

CONFIG="${CONFIG:-configs/cifar10_davidnet.yaml}"
PARALLEL="${PARALLEL:-2}"
SEEDS="${SEEDS:-3}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d data/cifar10 ]]; then
  echo "[warn] data/cifar10 not found. Run: python scripts/prepare_vision_data.py --dataset cifar10"
fi

DRY_FLAG=""
if [[ "$DRY_RUN" == "1" ]]; then
  DRY_FLAG="--dry-run"
fi

# (optimizer_name, comma-separated LR list)
MUON_LRS="0.005,0.01,0.02,0.04"
MUSCALE_LRS="${MUSCALE_LRS:-0.04,0.06,0.08,0.12}"
ADAM_LRS="0.001,0.003,0.01"

declare -a JOBS=(
  "muon                                ${MUON_LRS}"
  "muon_moonlight                      ${MUON_LRS}"
  "orscale_muon                        ${MUON_LRS}"
  "orscale_muon_wd                     ${MUON_LRS}"
  "orscale_muon_moonlight              ${MUON_LRS}"
  "orscale_muon_moonlight_calibrated   ${MUON_LRS}"
  "mutrust                             ${MUON_LRS}"
  "muscale                             ${MUSCALE_LRS}"
  "adamw                               ${ADAM_LRS}"
  "lamb                                ${ADAM_LRS}"
)

# Per-optimizer extra `--set` overrides passed through sweep.py.  Both
# Moonlight-shape variants use LARS/LAMB-style looser clip bounds:
#  - orscale_muon_moonlight: shape-constant denominator gives a wider
#    natural trust-ratio range; the tight default fires r_min~16% of
#    steps at the optimal LR on FineWeb.
#  - orscale_muon_moonlight_calibrated: auto-calibrated denominator sets
#    r̂(0)=1 per layer, so the natural operating range is wider still.
extra_set_for_opt() {
  case "$1" in
    orscale_muon_moonlight | orscale_muon_moonlight_calibrated)
      echo "optimizer.r_min=0.1 optimizer.r_max=5.0"
      ;;
    *)
      echo ""
      ;;
  esac
}

OPTIMIZERS="${OPTIMIZERS:-}"
declare -a SELECTED_JOBS=()
if [[ -n "$OPTIMIZERS" ]]; then
  # Accept comma- or space-separated names, e.g. "muon,adamw" or "muon adamw".
  normalized_optimizers="${OPTIMIZERS//,/ }"
  for requested_opt in $normalized_optimizers; do
    found=0
    for entry in "${JOBS[@]}"; do
      read -r opt_name _lrs <<< "$entry"
      if [[ "$opt_name" == "$requested_opt" ]]; then
        SELECTED_JOBS+=("$entry")
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "[error] unknown optimizer in OPTIMIZERS: $requested_opt" >&2
      echo "        valid choices: muon muon_moonlight orscale_muon orscale_muon_wd orscale_muon_moonlight orscale_muon_moonlight_calibrated mutrust muscale adamw lamb" >&2
      exit 1
    fi
  done
else
  SELECTED_JOBS=("${JOBS[@]}")
fi

if [[ "${#SELECTED_JOBS[@]}" -eq 0 ]]; then
  echo "[error] no optimizers selected." >&2
  exit 1
fi

echo "==============================================="
echo " OrScale CIFAR-10 sweep"
echo "   config     : $CONFIG"
echo "   parallel   : $PARALLEL"
echo "   seeds      : $SEEDS"
echo "   optimizers : ${OPTIMIZERS:-all}"
echo "   dry-run    : $DRY_RUN"
echo "==============================================="

for entry in "${SELECTED_JOBS[@]}"; do
  read -r OPT LRS <<< "$entry"
  echo
  echo ">>> Sweeping optimizer=${OPT}  lrs=[${LRS}]"

  # Per-variant fixed overrides (passed as singleton --sweep entries so they
  # propagate to every (lr, seed) cell as `key=value` overrides).  Built as a
  # space-separated string and only spliced into the cmd when non-empty (so
  # the `set -u` discipline does not fire on the empty-array expansion path,
  # which is unsafe on bash 3.2).
  EXTRA_SET="$(extra_set_for_opt "$OPT")"

  if [[ -n "$EXTRA_SET" ]]; then
    # shellcheck disable=SC2086
    python scripts/sweep.py \
      --script scripts/train_vision.py \
      --config "$CONFIG" \
      --sweep "optimizer.name=${OPT}" "optimizer.lr=${LRS}" $EXTRA_SET \
      --seeds "$SEEDS" \
      --parallel "$PARALLEL" \
      --no-stream \
      $DRY_FLAG
  else
    python scripts/sweep.py \
      --script scripts/train_vision.py \
      --config "$CONFIG" \
      --sweep "optimizer.name=${OPT}" "optimizer.lr=${LRS}" \
      --seeds "$SEEDS" \
      --parallel "$PARALLEL" \
      --no-stream \
      $DRY_FLAG
  fi
done

echo
echo "All sweeps dispatched. Per-run logs are under sweeps/<timestamp>/."
