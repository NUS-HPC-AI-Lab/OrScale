#!/usr/bin/env bash
# Sweep all 8 OrScale memo variants + AdamW + LAMB on FineWeb-Edu (LM) at the
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
#     run uses ~131K tokens/step by default (32 * 1024 * NPROC * grad_accum);
#     set TARGET_TOKENS_PER_STEP=262144 for ~262K tokens/step (e.g. muscale
#     PBS job parity with other cluster sweeps).
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
#   Muon grid          : {0.005, 0.01, 0.02, 0.04}   (muon, orscale_muon,
#                                                     orscale)
#   AdamW / LAMB grid  : {3e-4, 1e-3, 3e-3}          (adamw, lamb)
#   Moonlight grid     : {3e-4, 1e-3, 3e-3, 5e-3}    (muon_moonlight,
#                                                     orscale_muon_moonlight,
#                                                     orscale_lm)
#   muscale grid       : {1e-4, 3e-4, 5e-4, 1e-3}    (muscale only -- denser
#                                                     low-LR bracket per
#                                                     reports/fineweb_small/)
#   mutrust grid       : {0.005, 0.01, 0.02}         (mutrust)
#
# Per-variant trust-ratio clip overrides:
#   Default (set in YAML)              : r_min=0.5, r_max=1.5
#     This is the post-2026-04-22 tight clip used by the older OrScale
#     variants (orscale_muon, orscale, mutrust, muscale).  For
#     mutrust and muscale specifically, this clip is the only thing
#     keeping the optimizer from running at runaway effective LR -- their
#     raw ratio is O(1/lr) and saturates the clip on ~100% of steps no
#     matter what r_max is.
#   Moonlight-shape variants            : r_min=0.1, r_max=5.0
#     (orscale_muon_moonlight, orscale_lm)
#     These have a shape-constant or auto-calibrated denominator with a
#     wider natural operating range and benefit from LARS/LAMB-style
#     looser bounds.  The calibrated variant's denominator is auto-set
#     per layer so r̂(0)=1; the original Moonlight variant's r̂ runs in
#     [0.5, 0.74] at the optimal LR on FineWeb, with r_min=0.5 firing
#     ~16% of the time -- looser bounds remove that artefact while still
#     catching pathological steps.  The analytic LAMB convention is
#     [0, 10]; we tighten to [0.1, 5] because Muon's orthogonalization
#     already controls the direction.
#
# Why four grids instead of two (history):
#   Update 2026-04-22: The `0.2 * sqrt(max(m, n))` Moonlight shape-
#   normalization constant inflates the per-entry update magnitude by ~11x on
#   `small_125m`'s largest layers (see reports/fineweb_bump/). Sweeping the
#   two Moonlight-scaled variants on the Muon grid (0.005-0.04) produces a
#   sustained dip -> bump -> dip training-loss pattern at every LR, whereas
#   the same LRs on vanilla Muon are clean. Moonlight (arXiv:2502.16982
#   Sec 2.2) recommends reusing AdamW's LR with the shape constant; we widen
#   that slightly upward to bracket the onset of instability.
#
#   Update 2026-04-28 (post-fix sweep, see reports/fineweb_small/):
#   With r_max tightened from 10.0 to 1.5 and a global grad_clip_norm of 1.0,
#   the optima are: muon_moonlight @ lr=1e-3, orscale_muon_moonlight @ 3e-3,
#   mutrust @ 0.02, all converging within 0.02 nats of muon@0.02 (3.2111).
#   The Moonlight grid's lr=1e-2 still diverges (val 4.6 vs. 3.23) -- it is
#   replaced with 5e-3 to give a finer cell between 3e-3 and the divergence
#   line. mutrust@0.04 reaches a worse basin (val 3.73) with the trust-ratio
#   cap and grad-norm clip both saturated post-warmup, so it gets its own
#   3-cell grid that drops the bad upper edge.
#
#   Update 2026-04-29 (muscale FineWeb re-grid):
#   First muscale sweep on {3e-4, 1e-3, 3e-3, 5e-3} showed clean runs only on
#   the lower half; 3e-3/5e-3 dip-bump-diverge with grad clip saturated. Best
#   at 1e-3. muscale now uses a dedicated MUSCALE_LRS {1e-4, 3e-4, 5e-4, 1e-3}
#   (not the Moonlight grid shared with muon_moonlight / orscale_muon_moonlight).
#
#   Update 2026-04-28 (muscale added, superseded for LR grid by 2026-04-29):
#   `muscale` = mutrust trust ratio + Moonlight shape factor 0.2*sqrt(max(m,n)).
#   Its trust ratio is mathematically identical to mutrust's (RMS form just
#   cancels the sqrt(mn) factor), but its update is multiplied by the same
#   shape factor as muon_moonlight / orscale_muon_moonlight, so its effective
#   per-entry update RMS is in the Moonlight band. The first sweep used the
#   shared Moonlight LRs; the dedicated grid above reflects empirical results.
#
# Optimizers       : 7 Muon-family + AdamW + LAMB (same set as CIFAR sweep)
# Seeds per cell   : 1 (LM runs are expensive; bump to 2-3 for a final pass)
# Total runs       : (3 Muon * 4 + 2 Moonlight * 4 + 1 muscale * 4 + 1 Mutrust * 3 +
#                    2 Adam * 3) * 1 = 33 per config. With both pilot_25m +
#                    small_125m: 66 runs.
#
# Usage:
#   bash scripts/sweep_fineweb_small.sh                   # both configs
#   CONFIGS=configs/small_125m.yaml \
#     bash scripts/sweep_fineweb_small.sh                 # just small
#   NPROC=8 bash scripts/sweep_fineweb_small.sh           # use 8 GPUs/run
#   SEEDS=3 bash scripts/sweep_fineweb_small.sh           # 3 seeds per cell
#   DRY_RUN=1 bash scripts/sweep_fineweb_small.sh         # print only
#   TARGET_TOKENS_PER_STEP=262144 \
#     bash scripts/sweep_fineweb_small.sh                 # align with full-batch other sweeps (2x 131072)
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
TARGET_TOKENS_PER_STEP="${TARGET_TOKENS_PER_STEP:-131072}"
# 131072 = 2 * (batch * seq * nproc) with small_125m @ NPROC=2; use 262144
# (4x micro-step) to match other optimizers' effective batch in cluster jobs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d data/fineweb10B ]]; then
  echo "[warn] data/fineweb10B not found. Run: python scripts/prepare_data.py --version 10B"
fi

compute_grad_accum_for_target() {
  local config="$1"
  python - "$config" "$NPROC" "$TARGET_TOKENS_PER_STEP" <<'PY'
import sys

config_path, nproc_raw, target_raw = sys.argv[1:]
nproc = int(nproc_raw)
target = int(target_raw)

def read_section_int(path: str, section_name: str, key_name: str, default: int) -> int:
    section = None
    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
                section = line[:-1].strip()
                continue
            if section == section_name:
                stripped = line.strip()
                prefix = f"{key_name}:"
                if stripped.startswith(prefix):
                    value = stripped[len(prefix):].strip().strip("'\"")
                    return int(value)
    return default

seq_len = read_section_int(config_path, "model", "max_seq_len", 1024)
batch_size = read_section_int(config_path, "training", "batch_size", 32)
tokens_per_micro_step = batch_size * seq_len * nproc

if target % tokens_per_micro_step != 0:
    raise SystemExit(
        f"[error] TARGET_TOKENS_PER_STEP={target} is not divisible by "
        f"batch_size({batch_size}) * seq_len({seq_len}) * NPROC({nproc}) "
        f"= {tokens_per_micro_step} for {config_path}"
    )

grad_accum = target // tokens_per_micro_step
if grad_accum < 1:
    raise SystemExit(
        f"[error] TARGET_TOKENS_PER_STEP={target} is smaller than one "
        f"micro-step ({tokens_per_micro_step} tokens) for {config_path}"
    )

print(f"{grad_accum} {tokens_per_micro_step} {target}")
PY
}

# Per-family LR grids.
MUON_LRS=(0.005 0.01 0.02 0.04)
ADAM_LRS=(3e-4 1e-3 3e-3)
# Moonlight-scaled Muon variants: widen the AdamW grid one step upward (5e-3)
# to bracket the optimum found at lr=3e-3 in reports/fineweb_small/. The
# previous upper edge (1e-2) was confirmed to diverge under the post-fix
# defaults so it is dropped from the final grid.
MOONLIGHT_LRS=(3e-4 1e-3 3e-3 5e-3)
# muscale: denser bracket below 1e-3 (see reports/fineweb_small/ 2026-04-29).
MUSCALE_LRS=(1e-4 3e-4 5e-4 1e-3)
# mutrust uses the lower three cells of the Muon grid: lr=0.04 saturates both
# r_max=1.5 and the grad-norm clip post-warmup and converges to a worse basin
# (val 3.73 vs. 3.22 at the optimum), see reports/fineweb_small/.
MUTRUST_LRS=(0.005 0.01 0.02)

# (optimizer_name, lr_family) where lr_family selects MUON_LRS, ADAM_LRS,
# MOONLIGHT_LRS, MUSCALE_LRS, or MUTRUST_LRS.
declare -a JOBS=(
  "muon                                MUON"
  "muon_moonlight                      MOONLIGHT"
  "orscale_muon                        MUON"
  "orscale                             MUON"
  "orscale_muon_moonlight              MOONLIGHT"
  "orscale_lm                          MOONLIGHT"
  "mutrust                             MUTRUST"
  "muscale                             MUSCALE"
  "adamw                               ADAM"
  "lamb                                ADAM"
)

# Emit per-optimizer extra `--set` overrides (trust-ratio clip etc.).  Returns
# a space-separated string of `key=value` entries to be appended to the
# train.py --set list, or empty when the YAML defaults are correct.
extra_set_for_opt() {
  case "$1" in
    orscale_muon_moonlight | orscale_lm)
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
      read -r opt_name _fam <<< "$entry"
      if [[ "$opt_name" == "$requested_opt" ]]; then
        SELECTED_JOBS+=("$entry")
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "[error] unknown optimizer in OPTIMIZERS: $requested_opt" >&2
      echo "        valid choices: muon muon_moonlight orscale_muon orscale orscale_muon_moonlight orscale_lm mutrust muscale adamw lamb" >&2
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
    MUSCALE)   total=$((total + ${#MUSCALE_LRS[@]})) ;;
    MUTRUST)   total=$((total + ${#MUTRUST_LRS[@]})) ;;
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
echo "   target tokens/step : $TARGET_TOKENS_PER_STEP"
echo "   seeds     : $SEEDS (base=$SEED_BASE)"
echo "   muon lrs      : ${MUON_LRS[*]}"
echo "   adam lrs      : ${ADAM_LRS[*]}"
echo "   moonlight lrs : ${MOONLIGHT_LRS[*]}"
echo "   muscale lrs   : ${MUSCALE_LRS[*]}"
echo "   mutrust lrs   : ${MUTRUST_LRS[*]}"
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
  read -r GRAD_ACCUM_STEPS TOKENS_PER_MICRO_STEP EFFECTIVE_TOKENS_PER_STEP < <(
    compute_grad_accum_for_target "$CONFIG"
  )

  echo
  echo ">>> [${CFG_STEM}] fixed batch: micro-step=${TOKENS_PER_MICRO_STEP} tokens, grad_accum=${GRAD_ACCUM_STEPS}, effective=${EFFECTIVE_TOKENS_PER_STEP} tokens/step"

  for entry in "${SELECTED_JOBS[@]}"; do
    read -r OPT FAM <<< "$entry"
    case "$FAM" in
      MUON)      LRS=("${MUON_LRS[@]}") ;;
      ADAM)      LRS=("${ADAM_LRS[@]}") ;;
      MOONLIGHT) LRS=("${MOONLIGHT_LRS[@]}") ;;
      MUSCALE)   LRS=("${MUSCALE_LRS[@]}") ;;
      MUTRUST)   LRS=("${MUTRUST_LRS[@]}") ;;
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
          "training.grad_accum_steps=${GRAD_ACCUM_STEPS}"
        )

        # Append per-variant clip / hyperparameter overrides if any.
        EXTRA_SET="$(extra_set_for_opt "$OPT")"
        if [[ -n "$EXTRA_SET" ]]; then
          for kv in $EXTRA_SET; do
            cmd+=("$kv")
          done
        fi

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
