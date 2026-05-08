# OrScale

**Orthogonalised optimisation with layer-wise trust-ratio scaling.**

OrScale adds LARS/LAMB-style layer-wise magnitude control to Muon's orthogonalised matrix updates. The key rule is simple: the trust-ratio denominator should measure the Frobenius norm of the real parameter-space direction that will be subtracted from the weights.

Public repository: [NUS-HPC-AI-Lab/OrScale](https://github.com/NUS-HPC-AI-Lab/OrScale).

## Main Variants

The public release follows the naming used in the NeurIPS paper draft:

| Paper name | Config name | Intended use | Update denominator |
|---|---|---|---|
| **OrScale** | `orscale` | General matrix layers, vision experiments | `||lambda W + Q||_F` |
| **OrScale-LM** | `orscale_lm` | Language-model pre-training | `c_denom * ||lambda W + s Q||_F`, with `s = 0.2 * sqrt(max(m, n))` |

`OrScale` is the general recipe. `OrScale-LM` adds Moonlight's shape factor and a one-time per-layer calibration so each trust ratio starts at one, preserving learning-rate transfer for language models.

## Installation

```bash
python -m pip install -e .
```

Optional extras are split by workflow:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[data,vision,eval,analysis,wandb]"
```

For the all-in-one compatibility path:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Language-model smoke run:

```bash
python scripts/train.py --config configs/pilot_25m.yaml \
    --set optimizer.name=orscale_lm
```

CIFAR-10 / DavidNet run:

```bash
python scripts/train_vision.py --config configs/cifar10_davidnet.yaml \
    --set optimizer.name=orscale
```

The default configs use relative paths such as `data/fineweb10B/`, `data/cifar10/`, and `checkpoints/`. Override paths with `--set data.train_pattern=... data.val_pattern=... training.save_dir=...`.

W&B logging is opt-in. Set `logging.wandb_project` in the config or command-line overrides to enable it.

## Empirical Results

### CIFAR-10 / DavidNet

Best learning rate per optimizer, validation top-1 averaged over the last three of 24 epochs and then over three seeds:

| Rank | Optimizer | LR | Val top-1 |
|---:|---|---:|---:|
| 1 | **OrScale** | 0.02 | **94.05 +/- 0.08** |
| 2 | Muon + Moonlight | 0.01 | 93.75 +/- 0.17 |
| 3 | Muon | 0.04 | 93.70 +/- 0.14 |
| 4 | AdamW | 0.01 | 93.12 +/- 0.04 |
| 5 | LAMB | 0.01 | 92.40 +/- 0.20 |

### FineWeb-Edu Pre-Training

Final validation cross-entropy at four model scales. Lower is better.

| Scale | Compute | AdamW | Muon + Moonlight | OrScale-LM |
|---|---:|---:|---:|---:|
| 125M, 5.24B tokens | 0.046 PFD | 3.3721 | 3.2319 | **3.2120** |
| 399M, 8.92B tokens | 0.247 PFD | 2.9966 | **2.9183** | 2.9247 |
| 545M, 14.04B tokens | 0.531 PFD | 2.9235 | 2.8130 | **2.8049** |
| 1.1B, 28.54B tokens | 2.18 PFD | 2.7304 | 2.6360 | **2.6251** |

OrScale-LM beats AdamW at every scale from 125M to 1.1B and beats Muon + Moonlight at three of four scales. The 399M cell is within single-seed noise in the paper discussion.

## Data Preparation

FineWeb-Edu token shards:

```bash
python scripts/prepare_data.py --version 10B
```

CIFAR-10:

```bash
python scripts/prepare_vision_data.py --dataset cifar10
```

ImageNet expects the standard ImageFolder layout. See `scripts/prepare_vision_data.py` for tarball extraction support.

## Tests

```bash
pytest tests/ -v
```

On CPU-only machines without an OpenMP-capable compiler:

```bash
TORCH_COMPILE_DISABLE=1 pytest tests/ -v
```

## Repository Layout

```text
orscale/      Core optimizers, models, data loaders, trainers, eval, analysis
configs/      Example LM, vision, and scaling-law configs
scripts/      Training, data preparation, evaluation, and sweep entry points
tests/        Unit and smoke tests
```

Generated outputs under `results/`, `reports/`, checkpoints, datasets, W&B runs, and local logs are intentionally ignored.

## To Do

- More experiments of OrScale.
- TPU adaption.

## Citation

If you use OrScale in your research, please cite this repository and the associated paper when available. The repository includes `CITATION.cff` so GitHub can surface citation metadata.

## License

OrScale is released under the MIT License. See `LICENSE` for details.
