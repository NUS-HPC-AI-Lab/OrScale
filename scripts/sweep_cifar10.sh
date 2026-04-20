#!/usr/bin/env bash
# Sweep all 6 OrScale memo variants + AdamW + LAMB on CIFAR-10 / DavidNet.
#
# Hardware target  : 2 x A100-40G (single node, no DDP).
#   Each run uses 1 GPU, batch_size=512, matching the Muon blog reference.
#   sweep.py launches `--parallel 2` runs at a time, assigning CUDA_VISIBLE_DEVICES
#   per run (logical device 0 inside the child process maps to one physical GPU).
#
# Optimizers       : 6 Muon-family + AdamW + LAMB
# Per-family LR    : Muon family {0.005, 0.01, 0.02, 0.04}
#                    AdamW/LAMB  {1e-3, 3e-3, 1e-2}
# Seeds per cell   : 3
# Total runs       : (6 * 4 + 2 * 3) * 3 = 90
#
# Usage:
#   bash scripts/sweep_cifar10.sh                 # full sweep
#   DRY_RUN=1 bash scripts/sweep_cifar10.sh       # print runs without launching
#   PARALLEL=1 bash scripts/sweep_cifar10.sh      # force sequential (e.g. 1 GPU)
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
ADAM_LRS="0.001,0.003,0.01"

declare -a JOBS=(
  "muon                    ${MUON_LRS}"
  "muon_moonlight          ${MUON_LRS}"
  "orscale_muon            ${MUON_LRS}"
  "orscale_muon_wd         ${MUON_LRS}"
  "orscale_muon_moonlight  ${MUON_LRS}"
  "mutrust                 ${MUON_LRS}"
  "adamw                   ${ADAM_LRS}"
  "lamb                    ${ADAM_LRS}"
)

echo "==============================================="
echo " OrScale CIFAR-10 sweep"
echo "   config   : $CONFIG"
echo "   parallel : $PARALLEL"
echo "   seeds    : $SEEDS"
echo "   dry-run  : $DRY_RUN"
echo "==============================================="

for entry in "${JOBS[@]}"; do
  read -r OPT LRS <<< "$entry"
  echo
  echo ">>> Sweeping optimizer=${OPT}  lrs=[${LRS}]"
  python scripts/sweep.py \
    --script scripts/train_vision.py \
    --config "$CONFIG" \
    --sweep "optimizer.name=${OPT}" "optimizer.lr=${LRS}" \
    --seeds "$SEEDS" \
    --parallel "$PARALLEL" \
    --no-stream \
    $DRY_FLAG
done

echo
echo "All sweeps dispatched. Per-run logs are under sweeps/<timestamp>/."
