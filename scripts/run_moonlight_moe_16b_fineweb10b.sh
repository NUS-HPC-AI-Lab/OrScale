#!/usr/bin/env bash
# OrScale vs Moonlight Muon on the Moonlight-16B-A3B MoE shape.
#
# Default is DRY_RUN=1 so command derivation is safe. The config enables the
# repo's ZeRO-1-style optimizer-state sharding by default (`optimizer.zero_stage=1`):
# parameters/gradients stay replicated for Muon's matrix orthogonalization,
# while optimizer states are partitioned across the 8 H100 data-parallel ranks.
#
# Examples:
#   bash scripts/run_moonlight_moe_16b_fineweb10b.sh
#   OPTIMIZER=muon_moonlight bash scripts/run_moonlight_moe_16b_fineweb10b.sh
#   PYTHON=.venv/bin/python OPTIMIZER=orscale_lm bash scripts/run_moonlight_moe_16b_fineweb10b.sh
#   OPTIMIZER=orscale_lm DRY_RUN=0 TRAIN_PATTERN=/data/fineweb10B/fineweb_train_*.bin \
#     VAL_PATTERN=/data/fineweb10B/fineweb_val_*.bin \
#     bash scripts/run_moonlight_moe_16b_fineweb10b.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/moonlight_moe_16b_fineweb10b_compare.yaml}"
OPTIMIZER="${OPTIMIZER:-}"
PRESETS="${PRESETS:-${PRESET:-}}"
DRY_RUN="${DRY_RUN:-1}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
TRAIN_PATTERN="${TRAIN_PATTERN:-}"
VAL_PATTERN="${VAL_PATTERN:-}"
SAVE_DIR="${SAVE_DIR:-}"
WANDB_GROUP="${WANDB_GROUP:-moonlight_moe_16b_fineweb10b}"
ESTIMATE_PFLOPS_PER_SEC="${ESTIMATE_PFLOPS_PER_SEC:-}"
EXTRA_SET="${EXTRA_SET:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_TIMEOUT_MS="${TORCH_NCCL_TIMEOUT_MS:-1800000}"

cmd=(
  "$PYTHON"
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
echo " Moonlight MoE 16B FineWeb-10B optimizer comparison"
echo "   config         : $CONFIG"
echo "   optimizer      : ${OPTIMIZER:-all}"
echo "   presets        : ${PRESETS:-all}"
echo "   dry-run        : $DRY_RUN"
echo "   skip train     : $SKIP_TRAINING"
echo "   note           : optimizer.zero_stage=1 shards optimizer states across ranks"
echo " Memory/NCCL env (effective values):"
echo "   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "   TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
echo "   TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT"
echo "   TORCH_NCCL_TIMEOUT_MS=$TORCH_NCCL_TIMEOUT_MS"
echo "============================================================"
echo "${cmd[*]}"
echo

"${cmd[@]}"
