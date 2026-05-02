#!/usr/bin/env bash
# Strict Moonlight Table 2 scaling-law runs on one 8-GPU node.
#
# Examples:
#   DRY_RUN=1 OPTIMIZER=adamw bash scripts/run_moonlight_scaling_8gpu.sh
#   OPTIMIZER=orscale_muon_moonlight_calibrated bash scripts/run_moonlight_scaling_8gpu.sh
#   PRESETS=moonlight_399m,moonlight_545m DRY_RUN=1 bash scripts/run_moonlight_scaling_8gpu.sh
#   PRESETS="fineweb_small_125m moonlight_399m" OPTIMIZER=adamw bash scripts/run_moonlight_scaling_8gpu.sh
#
# Optional train.py overrides:
#   TRAIN_PATTERN=/data/fineweb10B/fineweb_train_*.bin
#   VAL_PATTERN=/data/fineweb10B/fineweb_val_*.bin
#   SAVE_DIR=/data/checkpoints/moonlight_scaling
#   WANDB_GROUP=moonlight_scaling_strict
#   ESTIMATE_PFLOPS_PER_SEC=0.65
#   EXTRA_SET="training.save_every=0 diagnostics.log_every=100"

set -euo pipefail

CONFIG="${CONFIG:-configs/scaling_law_moonlight_strict.yaml}"
OPTIMIZER="${OPTIMIZER:-}"
PRESETS="${PRESETS:-${PRESET:-}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
TRAIN_PATTERN="${TRAIN_PATTERN:-}"
VAL_PATTERN="${VAL_PATTERN:-}"
SAVE_DIR="${SAVE_DIR:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
ESTIMATE_PFLOPS_PER_SEC="${ESTIMATE_PFLOPS_PER_SEC:-}"
EXTRA_SET="${EXTRA_SET:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Runs training with the repo's conda env (see README / other scripts using `conda run -n orscale`).
CONDA_ENV="${CONDA_ENV:-orscale}"

cmd=(
  conda run -n "$CONDA_ENV" --no-capture-output
  python
  scripts/run_scaling_law.py
  --config "$CONFIG"
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ "$SKIP_TRAINING" == "1" ]]; then
  cmd+=(--skip-training)
fi

if [[ -n "$OPTIMIZER" ]]; then
  cmd+=(--only-optimizer "$OPTIMIZER")
fi

if [[ -n "$PRESETS" ]]; then
  cmd+=(--only-preset "$PRESETS")
fi

if [[ -n "$ESTIMATE_PFLOPS_PER_SEC" ]]; then
  cmd+=(--estimate-pflops-per-second "$ESTIMATE_PFLOPS_PER_SEC")
fi

extra_overrides=()
if [[ -n "$TRAIN_PATTERN" ]]; then
  extra_overrides+=("data.train_pattern=$TRAIN_PATTERN")
fi
if [[ -n "$VAL_PATTERN" ]]; then
  extra_overrides+=("data.val_pattern=$VAL_PATTERN")
fi
if [[ -n "$SAVE_DIR" ]]; then
  extra_overrides+=("training.save_dir=$SAVE_DIR")
fi
if [[ -n "$WANDB_GROUP" ]]; then
  extra_overrides+=("logging.wandb_group=$WANDB_GROUP")
fi
if [[ -n "$EXTRA_SET" ]]; then
  for kv in $EXTRA_SET; do
    extra_overrides+=("$kv")
  done
fi

if [[ "${#extra_overrides[@]}" -gt 0 ]]; then
  cmd+=(--set "${extra_overrides[@]}")
fi

echo "============================================================"
echo " Moonlight strict scaling run"
echo "   config    : $CONFIG"
echo "   optimizer : ${OPTIMIZER:-all}"
echo "   presets   : ${PRESETS:-all}"
echo "   dry-run   : $DRY_RUN"
echo "   skip train: $SKIP_TRAINING"
echo "============================================================"
echo "${cmd[*]}"
echo

"${cmd[@]}"
