#!/usr/bin/env bash
# Sweep all 6 OrScale memo variants + AdamW + LAMB on FineWeb-Edu (LM) at the
# "small" scale (configs/small_125m.yaml, optionally configs/pilot_25m.yaml).
#
# Hardware target  : 1 node x 4 A100 (single-node DDP, one run at a time).
#   Each run uses all 4 GPUs via `torchrun --nproc_per_node=4`. Runs are
#   executed sequentially (the LM jobs are too heavy to share a node with
#   another DDP run; this differs from sweep_cifar10.sh which co-locates
#   single-GPU runs in parallel).
#
# Reference: Moonlight ("Muon is Scalable for LLM Training", arXiv:2502.16982).
#   - Llama-style dense models, Muon family vs AdamW scaling-law sweep.
#   - Smallest scaling-law cell: 399M params, 12L / hidden 1536, 8.92B tokens,
#     LR=9.503e-4, batch size 96 (in 8K context = ~786K tokens/step). Our 125M
#     run uses ~131K tokens/step (32 * 1024 * 4 grad-accum), so the AdamW
#     optimum is expected to land in the same ~3e-4 to ~3e-3 band.
#   - Momentum 0.95, Newton-Schulz iters = 5 (paper Sec 2.2; matches
#     OrScale defaults).
#   - Weight decay: paper uses 0.1 in scaling-law and Moonlight pretraining;
#     this script keeps the YAML defaults (Muon family wd=0.01, AdamW wd=0.01)
#     so the comparison stays apples-to-apples with the existing memo runs.
#     Override with `--set optimizer.weight_decay=0.1` if you want to follow
#     the paper exactly.
#
# Per-family LR grids (split so each optimizer is swept around its known-good
# operating point given its effective per-entry update magnitude at nominal LR):
#   Muon grid          : {0.005, 0.01, 0.02, 0.04}   (Muon, mutrust)
#   AdamW / LAMB grid  : {3e-4, 1e-3, 3e-3}          (AdamW, LAMB)
#   Moonlight grid     : {3e-4, 1e-3, 3e-3, 1e-2}    (muon_moonlight,
#                                                     orscale_muon_moonlight)
#
# Why three grids instead of two (update 2026-04-22):
#   The `0.2 * sqrt(max(m, n))` Moonlight shape-normalization constant inflates
#   the per-entry update magnitude by ~11x on `small_125m`'s largest layers
#   (see reports/fineweb_bump/). Sweeping the two Moonlight-scaled variants on
#   the Muon grid (0.005-0.04) produces a sustained dip -> bump -> dip
#   training-loss pattern at every LR, whereas the same LRs on vanilla Muon
#   are clean. Moonlight (arXiv:2502.16982 Sec 2.2) recommends reusing AdamW's
#   LR with the shape constant; we widen that slightly (up to 1e-2) to bracket
#   the optimum and confirm instability returns above 1e-2.
#
# mutrust stays on the Muon grid: once r_max is tightened from 10 to ~1.5 the
# clipped trust ratio no longer amplifies the LR, so the operating point
# coincides with vanilla Muon again (see memo in reports/fineweb_bump/).
#
# Optimizers       : 6 Muon-family + AdamW + LAMB (same set as CIFAR sweep)
# Seeds per cell   : 1 (LM runs are expensive; bump to 2-3 for a final pass)
# Total runs       : (4 Muon * 4 + 2 Moonlight * 4 + 2 Adam * 3) * 1 = 30 per
#                    config. With both pilot_25m + small_125m: 60 runs.
#
# Usage:
#   bash scripts/sweep_fineweb_small.sh                   # both configs
#   CONFIGS=configs/small_125m.yaml \
#     bash scripts/sweep_fineweb_small.sh                 # just small
#   NPROC=8 bash scripts/sweep_fineweb_small.sh           # use 8 GPUs/run
#   SEEDS=3 bash scripts/sweep_fineweb_small.sh           # 3 seeds per cell
#   DRY_RUN=1 bash scripts/sweep_fineweb_small.sh         # print only
#   OPTIMIZERS="muon,muon_moonlight,orscale_muon" \
#     bash scripts/sweep_fineweb_small.sh                 # optimizer subset
#
# Notes:
#   - Run from the repo root.
#   - Make sure FineWeb-Edu shards are present:
#       python scripts/prepare_data.py --version 10B
#   - W&B run names are auto-built by train.py from
#     `<config_stem>-<opt>-lr<lr>-seed<seed>` and grouped by the YAML's
#     `logging.wandb_group`, so each cell shows up distinctly.
#   - Per-run stdout is tee'd to sweeps/fineweb_<timestamp>/<run>.log so you
#     can inspect failures without scrolling the parent shell.

set -euo pipefail

CONFIGS="${CONFIGS:-configs/pilot_25m.yaml configs/small_125m.yaml}"
NPROC="${NPROC:-4}"
SEEDS="${SEEDS:-1}"
SEED_BASE="${SEED_BASE:-42}"
DRY_RUN="${DRY_RUN:-0}"
RDZV_PORT="${RDZV_PORT:-29500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d data/fineweb10B ]]; then
  echo "[warn] data/fineweb10B not found. Run: python scripts/prepare_data.py --version 10B"
fi

# Per-family LR grids.
MUON_LRS=(0.005 0.01 0.02 0.04)
ADAM_LRS=(3e-4 1e-3 3e-3)
# Moonlight-scaled Muon variants: widen the AdamW grid by one step upward to
# bracket the onset of instability (see header comment).
MOONLIGHT_LRS=(3e-4 1e-3 3e-3 1e-2)

# (optimizer_name, lr_family) where lr_family selects MUON_LRS, ADAM_LRS, or
# MOONLIGHT_LRS.
declare -a JOBS=(
  "muon                    MUON"
  "muon_moonlight          MOONLIGHT"
  "orscale_muon            MUON"
  "orscale_muon_wd         MUON"
  "orscale_muon_moonlight  MOONLIGHT"
  "mutrust                 MUON"
  "adamw                   ADAM"
  "lamb                    ADAM"
)

OPTIMIZERS="${OPTIMIZERS:-}"
declare -a SELECTED_JOBS=()
if [[ -n "$OPTIMIZERS" ]]; then
  # Accept comma- or space-separated names, e.g. "muon,adamw" or "muon adamw".
  normalized_optimizers="${OPTIMIZERS//,/ }"
  for requested_opt in $normalized_optimizers; do
    found=0
    for entry in "${JOBS[@]}"; do
      read -r opt_name _fam <<< "$entry"
      if [[ "$opt_name" == "$requested_opt" ]]; then
        SELECTED_JOBS+=("$entry")
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "[error] unknown optimizer in OPTIMIZERS: $requested_opt" >&2
      echo "        valid choices: muon muon_moonlight orscale_muon orscale_muon_wd orscale_muon_moonlight mutrust adamw lamb" >&2
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

# Count total runs for the banner.
total=0
for entry in "${SELECTED_JOBS[@]}"; do
  read -r _OPT FAM <<< "$entry"
  case "$FAM" in
    MUON)      total=$((total + ${#MUON_LRS[@]})) ;;
    ADAM)      total=$((total + ${#ADAM_LRS[@]})) ;;
    MOONLIGHT) total=$((total + ${#MOONLIGHT_LRS[@]})) ;;
    *) echo "[error] unknown LR family: $FAM" >&2; exit 1 ;;
  esac
done
total=$((total * SEEDS))
n_cfg=0
for _ in $CONFIGS; do n_cfg=$((n_cfg + 1)); done
total=$((total * n_cfg))

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="sweeps/fineweb_${TIMESTAMP}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$SWEEP_DIR"
fi

echo "============================================================"
echo " OrScale FineWeb LM sweep (small)"
echo "   configs   : $CONFIGS"
echo "   nproc     : $NPROC (torchrun, sequential DDP runs)"
echo "   seeds     : $SEEDS (base=$SEED_BASE)"
echo "   muon lrs      : ${MUON_LRS[*]}"
echo "   adam lrs      : ${ADAM_LRS[*]}"
echo "   moonlight lrs : ${MOONLIGHT_LRS[*]}"
echo "   optimizers: ${OPTIMIZERS:-all}"
echo "   dry-run   : $DRY_RUN"
echo "   total     : $total runs"
echo "   log dir   : $SWEEP_DIR"
echo "============================================================"

run_idx=0
for CONFIG in $CONFIGS; do
  if [[ ! -f "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
  fi
  CFG_STEM="$(basename "${CONFIG%.yaml}")"

  for entry in "${SELECTED_JOBS[@]}"; do
    read -r OPT FAM <<< "$entry"
    case "$FAM" in
      MUON)      LRS=("${MUON_LRS[@]}") ;;
      ADAM)      LRS=("${ADAM_LRS[@]}") ;;
      MOONLIGHT) LRS=("${MOONLIGHT_LRS[@]}") ;;
    esac

    echo
    echo ">>> [${CFG_STEM}] optimizer=${OPT}  lrs=[${LRS[*]}]"

    for LR in "${LRS[@]}"; do
      for ((S = 0; S < SEEDS; S++)); do
        SEED=$((SEED_BASE + S))
        run_idx=$((run_idx + 1))
        RUN_NAME="${CFG_STEM}-${OPT}-lr${LR}-seed${SEED}"
        LOG_PATH="${SWEEP_DIR}/${RUN_NAME}.log"

        echo "  --- Run ${run_idx}/${total}: ${RUN_NAME} ---"

        cmd=(
          torchrun
          --standalone
          --nproc_per_node="${NPROC}"
          --rdzv_backend=c10d
          --rdzv_endpoint="localhost:${RDZV_PORT}"
          scripts/train.py
          --config "${CONFIG}"
          --set
          "optimizer.name=${OPT}"
          "optimizer.lr=${LR}"
          "training.seed=${SEED}"
        )

        if [[ "$DRY_RUN" == "1" ]]; then
          echo "      [dry-run] ${cmd[*]}"
          continue
        fi

        echo "      log: ${LOG_PATH}"
        if ! "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"; then
          echo "      FAILED (see ${LOG_PATH}); continuing sweep." >&2
        fi
      done
    done
  done
done

echo
echo "Sweep complete. Per-run logs under ${SWEEP_DIR}/"
