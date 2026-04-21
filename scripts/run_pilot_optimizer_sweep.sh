#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/pilot_25m.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-1}"
PARALLEL="${PARALLEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
OPTIMIZER_SWEEP="${OPTIMIZER_SWEEP:-optimizer.name=adamw,lamb,muon,muon_moonlight,orscale_original,mutrust,muscale,muscale_alpha}"

usage() {
  cat <<'EOF'
Run the pilot-level optimizer sweep.

Usage:
  scripts/run_pilot_optimizer_sweep.sh [extra sweep specs...]

Examples:
  scripts/run_pilot_optimizer_sweep.sh
  scripts/run_pilot_optimizer_sweep.sh optimizer.lr=0.01,0.02,0.05
  scripts/run_pilot_optimizer_sweep.sh training.max_steps=2000 optimizer.lr=0.01
  SEEDS=3 PARALLEL=2 DRY_RUN=1 scripts/run_pilot_optimizer_sweep.sh optimizer.lr=0.02

Environment overrides:
  CONFIG           Base config path (default: configs/pilot_25m.yaml)
  PYTHON_BIN       Python executable (default: python)
  SEEDS            Number of seeds per combo (default: 1)
  PARALLEL         Max parallel runs passed to sweep.py (default: 1)
  DRY_RUN          Set to 1 to pass --dry-run
  OPTIMIZER_SWEEP  Optimizer sweep spec passed to sweep.py
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

cmd=(
  "${PYTHON_BIN}" scripts/sweep.py
  --config "${CONFIG}"
  --sweep "${OPTIMIZER_SWEEP}"
)

if (($# > 0)); then
  cmd+=("$@")
fi

cmd+=(
  --seeds "${SEEDS}"
  --parallel "${PARALLEL}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
