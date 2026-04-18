# OrScale

**Orthogonalized updates with layer-wise scaling for language model training.**

OrScale combines Muon's orthogonalized update direction (Newton-Schulz polar factor) with a redesigned layer-wise trust ratio for update magnitude control. This repository implements the optimizer family and training infrastructure for the OrScale research project.

## Optimizer Variants

Every variant that applies "Moonlight shape normalization" uses the full Moonlight RMS-matching constant `0.2 * sqrt(max(m, n))` (arXiv:2502.16982).

| Variant | Momentum | Trust Ratio Denom | Shape Norm | Key Feature |
|---------|----------|-------------------|------------|-------------|
| **Muon** | Nesterov | — | — | Baseline orthogonalized optimizer |
| **Muon + Moonlight** | Nesterov | — | 0.2·sqrt(max(m,n)) | Static shape normalization |
| **OrScale-original** | EMA | \|\|Q\|\|_F | — | Original proposal (degenerate denom) |
| **OrScale-Muon-Moonlight** | Nesterov | \|\|λW + 0.2·sqrt(max(m,n))·Q\|\|_F | 0.2·sqrt(max(m,n)) | Dynamic trust ratio + coupled WD on Moonlight |
| **MuTrust** | Nesterov | \|\|M_hat\|\|_F | — | Minimal fix: raw momentum denom |
| **MuScale** | Nesterov | RMS(M_hat) | 0.2·sqrt(max(m,n)) | Width-invariant trust ratio |
| **MuScale-alpha** | Nesterov | RMS(M_hat)^alpha | 0.2·sqrt(max(m,n)) | Partial trust ratio (best candidate) |

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

### Vision data (CIFAR-10 / ImageNet)

The vision entry point expects CIFAR-10 in torchvision's standard layout and ImageNet-1K in the `ImageFolder` `train/<wnid>/*.JPEG` + `val/<wnid>/*.JPEG` layout. Use `scripts/prepare_vision_data.py`:

```bash
# CIFAR-10: fully automatic download via torchvision (~170 MB).
python scripts/prepare_vision_data.py --dataset cifar10

# ImageNet-1K: drop the three ILSVRC2012 tarballs into one directory, then:
python scripts/prepare_vision_data.py --dataset imagenet \
    --src /path/to/imagenet_tars \
    --out /data/imagenet
```

For ImageNet you must have downloaded `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`, and `ILSVRC2012_devkit_t12.tar.gz` from [image-net.org](https://image-net.org) (account required). The script unpacks the 1,000 per-class inner tarballs for `train/`, and uses the devkit ground truth to reorganize the flat val tar into `val/<wnid>/`.

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

Sweep runs keep the configured `wandb_group` (for example `pilot_25m`) and now get readable W&B names such as `pilot_25m-muon-lr0.01-seed42`, making optimizer and seed comparisons easy to filter in the dashboard.

## Vision experiments (CIFAR-10, ImageNet)

The Muon family is also applied to convolutional networks in the LAMB (Table 3/5/6) and Muon-post papers. Use `scripts/train_vision.py` to reproduce these experiments.

### CIFAR-10 / DavidNet

Single GPU:

```bash
python scripts/train_vision.py --config configs/cifar10_davidnet.yaml
```

Multi-GPU DDP:

```bash
torchrun --nproc_per_node=4 scripts/train_vision.py \
    --config configs/cifar10_davidnet.yaml
```

### ImageNet / ResNet-50 (large-batch, multi-node)

Reproduces LAMB's Table 5 with the Muon/OrScale family instead of LAMB itself.

```bash
# 8 nodes x 8 GPUs = 64 GPUs, global batch 16384
torchrun \
    --nnodes=8 --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train_vision.py \
    --config configs/imagenet_resnet50_bs16k.yaml

# 16 nodes x 8 GPUs = 128 GPUs, global batch 32768, polynomial LR decay
torchrun \
    --nnodes=16 --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train_vision.py \
    --config configs/imagenet_resnet50_bs32k.yaml
```

Both ImageNet configs enable `sqrt_lr_scaling` (LAMB's square-root LR rule, Sec 4.3): the config's base `optimizer.lr` is tuned at `sqrt_lr_scaling_base_batch` (default 512) and multiplied by `sqrt(global_batch / base_batch)` at runtime. For ResNet-50 the stem conv and final classifier `fc` are kept on AdamW; all hidden Conv2d weights are routed to the Muon family (flattened internally to `[C_out, C_in*kH*kW]`).

The new LR schedules added for these runs are:

- `cosine` -- linear warmup + cosine decay (default).
- `poly`   -- `(1 - t/T)^power` decay (LAMB BERT / large-batch runs).
- `step`   -- linear-epoch warmup + step decay at milestones (Goyal et al. ImageNet schedule, LAMB Table 5).

## Downstream LM evaluation

After running an LM training job, evaluate the final checkpoint on the Moonlight / Muon-post downstream suite (HellaSwag, MMLU, TriviaQA, GSM8K, MBPP, HumanEval, ...):

```bash
python scripts/eval_downstream.py \
    --checkpoint checkpoints/pilot_25m/step_005000.pt \
    --tasks hellaswag,mmlu,gsm8k \
    --batch-size 8
```

This uses EleutherAI's `lm-evaluation-harness` (`pip install lm-eval`) under the hood. The adapter module `orscale.eval.downstream` wraps `GPT` with loglikelihood + generate hooks so it plugs into `lm-eval` directly. Add `downstream_eval_every` to your training config to run this evaluation periodically during training.

## Scaling-law driver

Reproduce Moonlight Figure 3 / Table 3 (Muon vs AdamW scaling law) by sweeping (model size, token budget) pairs and fitting `L(C) = A * C^alpha`:

```bash
python scripts/run_scaling_law.py --config configs/scaling_law.yaml
```

This launches a training run for each `(preset, tokens, optimizer)` combination, writes `scaling_law.csv`, and on completion fits the power law via `scipy.optimize.curve_fit` and saves `scaling_law.png` + `scaling_law_fits.json`. New Moonlight-sized GPT presets `xs_400m`, `s_550m`, `m_800m`, `l_1_1b`, and `xl_1_5b` are available in `orscale.model.gpt.PRESET_CONFIGS` for this sweep.

## Running Tests

```bash
pytest tests/ -v
```

The test suite includes:

- **test_newton_schulz.py** -- Verifies NS5 output matches SVD polar factor, correct Frobenius norms, batched operation.
- **test_orscale.py** -- All four variants reduce loss, trust ratio clipping, diagnostics populated.
- **test_degeneration.py** -- Critical correctness: MuScale(r=1) = Muon+Moonlight, MuScale-alpha(a=0) = Muon+Moonlight, MuScale-alpha(a=1) = MuScale.
- **test_vision_optim.py** -- Muon / OrScale handle 4D Conv2d weights via flatten round-trip, `_split_params` routes conv weights to the matrix group, one step reduces CE on a tiny ConvNet.
- **test_downstream_eval.py** -- `lm-eval` adapter smoke test on a tiny GPT (`tasks=["hellaswag"], limit=8`).
- **test_scaling_law.py** -- Power-law fitting recovers the exponent on synthetic `L = 3 * C^-0.05` data.

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
    gpt.py                # Configurable GPT-2/LLaMA model (tiny/small/medium/400M-1.5B presets)
    vision.py             # DavidNet, PreActResNet20, ResNet-50 (torchvision wrapper)
  data/
    loader.py             # .bin shard loader + HuggingFace fallback (LM)
    vision.py             # CIFAR-10 + ImageNet loaders with DDP samplers
  diagnostics/
    logger.py             # Per-layer metric collection + W&B logging
  training/
    trainer.py            # LM training loop (single-GPU + DDP)
    vision_trainer.py     # Vision training loop with epoch-based scheduling + top-k metrics
    scheduler.py          # Cosine / polynomial / step-decay LR schedules + sqrt LR scaling
  eval/
    downstream.py         # lm-evaluation-harness adapter
  analysis/
    scaling_law.py        # Power-law fitting + PFLOP/s-day helpers
  utils/
    distributed.py        # DDP setup helpers
configs/
  pilot_25m.yaml                    # ~25M param pilot experiments
  small_125m.yaml                   # ~125M param core experiments
  medium_350m.yaml                  # ~350M param scaling experiments
  cifar10_davidnet.yaml             # CIFAR-10 + DavidNet (LAMB Table 6 / Muon-post)
  imagenet_resnet50_bs16k.yaml      # ImageNet + ResNet-50, global bs 16K (LAMB Table 5)
  imagenet_resnet50_bs32k.yaml      # ImageNet + ResNet-50, global bs 32K, polynomial decay
  scaling_law.yaml                  # Moonlight-style (N, D) sweep
scripts/
  prepare_data.py         # Download & tokenize FineWeb into .bin shards
  prepare_vision_data.py  # Download CIFAR-10 / extract ImageNet-1K tarballs
  train.py                # LM training entry point
  train_vision.py         # Vision training entry point
  eval_downstream.py      # Run lm-eval on a saved checkpoint
  run_scaling_law.py      # Moonlight-style scaling-law sweep
  sweep.py                # HP sweep launcher
tests/
  test_newton_schulz.py
  test_orscale.py
  test_degeneration.py
  test_vision_optim.py
  test_downstream_eval.py
  test_scaling_law.py
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
