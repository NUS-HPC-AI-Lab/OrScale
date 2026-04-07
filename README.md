# OrScale

**Orthogonalized updates with layer-wise scaling for language model training.**

OrScale combines Muon's orthogonalized update direction (Newton-Schulz polar factor) with a redesigned layer-wise trust ratio for update magnitude control. This repository implements the optimizer family and training infrastructure for the OrScale research project.

## Optimizer Variants

| Variant | Momentum | Trust Ratio Denom | Shape Norm | Key Feature |
|---------|----------|-------------------|------------|-------------|
| **Muon** | Nesterov | — | — | Baseline orthogonalized optimizer |
| **Muon + Moonlight** | Nesterov | — | sqrt(max(m,n)) | Static shape normalization |
| **OrScale-original** | EMA | \|\|Q\|\|_F | — | Original proposal (degenerate denom) |
| **MuTrust** | Nesterov | \|\|M_hat\|\|_F | — | Minimal fix: raw momentum denom |
| **MuScale** | Nesterov | RMS(M_hat) | sqrt(max(m,n)) | Width-invariant trust ratio |
| **MuScale-alpha** | Nesterov | RMS(M_hat)^alpha | sqrt(max(m,n)) | Partial trust ratio (best candidate) |

Additional baselines: **AdamW**, **LAMB**.

## Setup

```bash
pip install -r requirements.txt
```

### Data

The training scripts expect pre-tokenized `.bin` files (GPT-2 tokenizer, uint16). To download and shard the FineWeb-Edu 10B dataset:

```bash
python scripts/prepare_data.py --version 10B
```

This writes shards to `data/fineweb10B/`. Use `--version 100B` for the larger subset, or `--out <dir>` to change the output location.

Alternatively, the data loader supports HuggingFace datasets as a fallback (slower, downloads on the fly).

## Quick Start

### Single GPU

```bash
python scripts/train.py --config configs/pilot_25m.yaml
```

### Override parameters on the command line

```bash
python scripts/train.py --config configs/pilot_25m.yaml \
    --set optimizer.name=muon optimizer.lr=0.01 training.max_steps=2000
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/small_125m.yaml
```

### Hyperparameter Sweep

```bash
# Sweep over optimizers and learning rates
python scripts/sweep.py --config configs/pilot_25m.yaml \
    --sweep optimizer.name=muon,muscale_alpha optimizer.lr=0.01,0.02,0.05 \
    --seeds 3

# Dry run to see what would be launched
python scripts/sweep.py --config configs/pilot_25m.yaml \
    --sweep optimizer.name=adamw,muon,mutrust,muscale,muscale_alpha \
    --seeds 1 --dry-run
```

## Running Tests

```bash
pytest tests/ -v
```

The test suite includes:

- **test_newton_schulz.py** -- Verifies NS5 output matches SVD polar factor, correct Frobenius norms, batched operation.
- **test_orscale.py** -- All four variants reduce loss, trust ratio clipping, diagnostics populated.
- **test_degeneration.py** -- Critical correctness: MuScale(r=1) = Muon+Moonlight, MuScale-alpha(a=0) = Muon+Moonlight, MuScale-alpha(a=1) = MuScale.

## Project Structure

```
orscale/
  optim/
    newton_schulz.py      # Polar Express orthogonalization (5-iter Newton-Schulz)
    muon.py               # Muon baseline optimizer
    orscale_optimizer.py  # OrScale family (all 4 variants in one class)
    lamb.py               # LAMB optimizer baseline
    __init__.py           # build_optimizer() factory
  model/
    gpt.py                # Configurable GPT-2/LLaMA model (tiny/small/medium presets)
  data/
    loader.py             # .bin shard loader + HuggingFace fallback
  diagnostics/
    logger.py             # Per-layer metric collection + W&B logging
  training/
    trainer.py            # Training loop (single-GPU + DDP)
    scheduler.py          # Cosine LR with warmup
  utils/
    distributed.py        # DDP setup helpers
configs/
  pilot_25m.yaml          # ~25M param pilot experiments
  small_125m.yaml         # ~125M param core experiments
  medium_350m.yaml        # ~350M param scaling experiments
scripts/
  prepare_data.py         # Download & tokenize FineWeb into .bin shards
  train.py                # Main training entry point
  sweep.py                # HP sweep launcher
tests/
  test_newton_schulz.py
  test_orscale.py
  test_degeneration.py
```

## Experiment Configs

| Config | Model | Params | Steps | Batch Size | Purpose |
|--------|-------|--------|-------|------------|---------|
| `pilot_25m` | tiny (6L/384d) | ~25M | 5,000 | 64 | Sanity checks, debug |
| `small_125m` | small (12L/768d) | ~125M | 20,000 | 32x4 accum | Core comparison |
| `medium_350m` | medium (24L/1024d) | ~350M | 40,000 | 16x8 accum | Scaling validation |

## Key Design Decisions

1. **Single OrScale class**: All four variants share one `OrScaleOptimizer` class, differing only in config flags. This prevents implementation divergence and makes ablations trivial.

2. **Muon-family optimizer factory**: `build_optimizer()` automatically splits model parameters into matrix (2D) and non-matrix groups, applying the Muon-family optimizer to matrices and AdamW to the rest.

3. **Diagnostic hooks**: Each optimizer exposes a `_diagnostics` dict with per-layer metrics (trust ratios, norms, clipping status) that the `DiagnosticLogger` reads after each step.

4. **Degeneration tests**: The test suite verifies that MuScale with constant trust ratio degenerates exactly to Muon + Moonlight, ensuring implementation correctness.
