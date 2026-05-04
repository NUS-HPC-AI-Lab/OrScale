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

# --- Memory & NCCL stability env (critical for 545m+ on 96 GB H20-3E) ---
# expandable_segments: lets the CUDA caching allocator grow segments in place
#   instead of needing contiguous chunks. Eliminates the fragmentation-induced
#   OOM/hang flakiness from Newton-Schulz dynamic allocations + torch.compile
#   when running near the memory ceiling. Available since PyTorch 2.1.
# max_split_size_mb: prevents the allocator from splitting large blocks into
#   small ones that later can't be coalesced.
# NCCL settings: surface hangs as errors instead of silently waiting forever
#   when one rank stalls in a long opt.step or compile pass while others wait
#   at an all-reduce barrier (this is the "stuck at step 30" failure mode).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
# Generous watchdog so a slow first opt.step (NS compile + alloc) doesn't
# trip a false-positive timeout, but real hangs still raise within 30 min.
export TORCH_NCCL_TIMEOUT_MS="${TORCH_NCCL_TIMEOUT_MS:-1800000}"

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
