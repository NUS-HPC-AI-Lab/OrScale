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
# Per-family LR (split grids around each family's known-good operating point):
#   Muon family   : {0.005, 0.01, 0.02, 0.04}   (small_125m.yaml uses 0.02)
#   AdamW / LAMB  : {3e-4, 1e-3, 3e-3}          (Moonlight Table 2 ~9.5e-4)
#
# Note on `muon_moonlight` / `orscale_muon_moonlight`: the Moonlight paper
# argues that with the `0.2 * sqrt(max(m,n))` shape normalization, Muon should
# reuse AdamW's LR (~1e-3 here). The repo's existing configs nonetheless place
# all six Muon variants near `lr=0.02`, so we sweep them on the Muon grid for
# consistency. If a Moonlight variant prefers the lower end of {0.005}, that
# is still a meaningful signal worth confirming with a follow-up sweep at
# AdamW-scale LRs.
#
# Optimizers       : 6 Muon-family + AdamW + LAMB (same set as CIFAR sweep)
# Seeds per cell   : 1 (LM runs are expensive; bump to 2-3 for a final pass)
# Total runs       : (6 * 4 + 2 * 3) * 1 = 30 per config
#                    With both pilot_25m + small_125m: 60 runs
#
# Usage:
#   bash scripts/sweep_fineweb_small.sh                   # both configs
#   CONFIGS=configs/small_125m.yaml \
#     bash scripts/sweep_fineweb_small.sh                 # just small
#   NPROC=8 bash scripts/sweep_fineweb_small.sh           # use 8 GPUs/run
#   SEEDS=3 bash scripts/sweep_fineweb_small.sh           # 3 seeds per cell
#   DRY_RUN=1 bash scripts/sweep_fineweb_small.sh         # print only
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

# (optimizer_name, lr_family) where lr_family selects MUON_LRS or ADAM_LRS.
declare -a JOBS=(
  "muon                    MUON"
  "muon_moonlight          MUON"
  "orscale_muon            MUON"
  "orscale_muon_wd         MUON"
  "orscale_muon_moonlight  MUON"
  "mutrust                 MUON"
  "adamw                   ADAM"
  "lamb                    ADAM"
)

# Count total runs for the banner.
total=0
for entry in "${JOBS[@]}"; do
  read -r _OPT FAM <<< "$entry"
  if [[ "$FAM" == "MUON" ]]; then
    total=$((total + ${#MUON_LRS[@]}))
  else
    total=$((total + ${#ADAM_LRS[@]}))
  fi
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
echo "   muon lrs  : ${MUON_LRS[*]}"
echo "   adam lrs  : ${ADAM_LRS[*]}"
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

  for entry in "${JOBS[@]}"; do
    read -r OPT FAM <<< "$entry"
    if [[ "$FAM" == "MUON" ]]; then
      LRS=("${MUON_LRS[@]}")
    else
      LRS=("${ADAM_LRS[@]}")
    fi

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
