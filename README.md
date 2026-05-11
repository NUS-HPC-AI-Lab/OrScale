# OrScale

**Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling**

[![arXiv](https://img.shields.io/badge/arXiv-2605.07815-b31b1b.svg)](https://arxiv.org/abs/2605.07815)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

OrScale equips Muon's orthogonalised matrix update with LARS/LAMB-style
layer-wise magnitude control. The recipe rests on a single design principle:
**the trust-ratio denominator must measure the Frobenius norm of the
parameter-space direction the optimizer is about to subtract from the
weights.** The principle yields a unified, parameter-light algorithm with two
specialisations — `OrScale` (general / vision) and `OrScale-LM` (language
modelling) — and rules out three superficially natural Muon–LAMB hybrids that
fail in practice through degenerate denominators, clip saturation, or
weight-norm runaway.

> **Paper:** [Lou & You, 2026 — *OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling*](https://arxiv.org/abs/2605.07815) (arXiv:2605.07815).
>
> **Code:** [NUS-HPC-AI-Lab/OrScale](https://github.com/NUS-HPC-AI-Lab/OrScale).

---

## Highlights

- **Algorithm.** A drop-in extension of Muon: keep the orthogonalised
  direction $Q_\ell = \mathrm{NS}_k(\widetilde M_\ell)$, scale it by a clipped
  layer-wise trust ratio, and couple weight decay into the trust-ratio-scaled
  step. One fp32 scalar per layer for the LM variant; below 1% wall-clock
  overhead vs. Muon.
- **Theory.** A nuclear-norm $O(1/\sqrt{T})$ nonconvex convergence guarantee, a
  layer-adaptive descent constant $\kappa_{\mathrm{eff}} > 1$ under measurable
  layer heterogeneity, and a strict separation from raw-momentum
  Muon–trust-ratio variants under empirically verified clip saturation.
- **Empirics.** OrScale ranks first on CIFAR-10 / DavidNet across three seeds,
  and OrScale-LM beats Muon + Moonlight on FineWeb-Edu pre-training at three of
  four scales (125M → 1.1B parameters) and beats AdamW at every scale.
- **Reproducibility.** Single-command training entry points, deterministic
  configs, and shipped sweep scripts; the public results in this repository
  match the paper's tables and figures.

## Method at a Glance

Both variants share Muon's front end (Nesterov-lookahead momentum followed by
$k$ Newton–Schulz iterations to obtain the polar factor $Q_\ell$) and apply a
clipped trust-ratio multiplier. With weight $W_\ell \in \mathbb{R}^{m_\ell \times n_\ell}$,
weight decay $\lambda$, shape factor $s_\ell$, and per-layer calibration
constant $c_{\mathrm{denom},\ell}$:

$$
D_{\ell,t} \;=\; \lambda W_{\ell,t} + s_\ell\, Q_{\ell,t},
\qquad
r_{\ell,t} \;=\; \frac{\lVert W_{\ell,t} \rVert_F}{c_{\mathrm{denom},\ell}\,\lVert D_{\ell,t} \rVert_F + \varepsilon},
$$

$$
W_{\ell,t+1} \;=\; W_{\ell,t} \;-\; \eta_t\,\mathrm{clip}(r_{\ell,t},\,r_{\min},\,r_{\max})\,D_{\ell,t}.
$$

The two recommended specialisations:

| Variant | Config name | Intended use | Shape factor $s_\ell$ | Calibration $c_{\mathrm{denom},\ell}$ |
|---|---|---|---|---|
| **OrScale** | `orscale` | General matrix layers, vision experiments | $1$ | $1$ |
| **OrScale-LM** | `orscale_lm` | Language-model pre-training | $0.2\sqrt{\max(m_\ell, n_\ell)}$ | Set once at the first non-zero step so $r_{\ell,0} = 1$ |

`OrScale-LM` adopts the Moonlight shape factor and a one-time per-layer
calibration that anchors every trust ratio at one, propagating learning-rate
transfer from AdamW → Muon + Moonlight → OrScale-LM without an extra sweep.

For the full algorithm, theoretical statements, and design-space analysis
(including the failure modes that this principle rules out), see the
[paper](https://arxiv.org/abs/2605.07815).

## Installation

OrScale targets PyTorch ≥ 2.0 and Python ≥ 3.10.

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

The default configs use relative paths such as `data/fineweb10B/`,
`data/cifar10/`, and `checkpoints/`. Override paths with
`--set data.train_pattern=... data.val_pattern=... training.save_dir=...`.

W&B logging is opt-in. Set `logging.wandb_project` in the config or via
command-line overrides to enable it.

## Empirical Results

### CIFAR-10 / DavidNet

Best learning rate per optimizer; validation top-1 averaged over the last three
of 24 epochs, then over three seeds ($\pm 1\sigma$).

| Rank | Optimizer | LR | Val top-1 (%) |
|---:|---|---:|---:|
| 1 | **OrScale** (ours) | 0.02 | **94.05 ± 0.08** |
| 2 | Muon + Moonlight | 0.01 | 93.75 ± 0.17 |
| 3 | Muon | 0.04 | 93.70 ± 0.14 |
| 4 | AdamW | 0.01 | 93.12 ± 0.04 |
| 5 | LAMB | 0.01 | 92.40 ± 0.20 |

OrScale improves Muon by **+0.35 points** and Muon + Moonlight by
**+0.30 points**, while LAMB — the standard trust-ratio baseline — trails by
**1.65 points**, confirming that a direct LAMB-style port to Muon is not
competitive without the design principle above.

### FineWeb-Edu Pre-Training

Final validation cross-entropy at four model scales spanning a 48× compute
range. Lower is better; **bold** marks the best optimizer at each scale.
Compute $C = 6ND$ in PFLOP-days (Kaplan estimate).

| Scale | Compute (PFD) | AdamW | Muon + Moonlight | **OrScale-LM** (ours) |
|---|---:|---:|---:|---:|
| 125M, 5.24B tok | 0.046 | 3.3721 | 3.2319 | **3.2120** |
| 399M, 8.92B tok | 0.247 | 2.9966 | **2.9183** | 2.9247 |
| 545M, 14.04B tok | 0.531 | 2.9235 | 2.8130 | **2.8049** |
| 1.1B, 28.54B tok | 2.18 | 2.7304 | 2.6360 | **2.6251** |

OrScale-LM beats AdamW at every scale from 125M to 1.1B and beats
Muon + Moonlight at three of four scales; the 399M cell is a tie within
single-seed noise. The fitted Kaplan-style scaling-law exponents are
$\alpha = -0.054$ (AdamW), $-0.053$ (Muon + Moonlight), and $-0.052$
(OrScale-LM); the OrScale-LM advantage is approximately preserved across the
swept compute range.

## Data Preparation

FineWeb-Edu token shards:

```bash
python scripts/prepare_data.py --version 10B
```

CIFAR-10:

```bash
python scripts/prepare_vision_data.py --dataset cifar10
```

ImageNet expects the standard `ImageFolder` layout. See
`scripts/prepare_vision_data.py` for tarball extraction support.

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

Generated outputs under `results/`, `reports/`, checkpoints, datasets,
W&B runs, and local logs are intentionally git-ignored.

## Roadmap

- Larger-scale empirical evaluation of OrScale on additional vision and
  language benchmarks.
- TPU adaptation of the orthogonalised front end and trust-ratio computation.
- Integration with attention stabilisers (e.g. MuonClip) and very-large-batch
  training regimes where layer-wise magnitude control is most pronounced.

## Citation

If you use OrScale in your research, please cite the paper:

```bibtex
@misc{lou2026orscaleorthogonalisedoptimizationlayerwise,
      title={OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling},
      author={Yuxuan Lou and Yang You},
      year={2026},
      eprint={2605.07815},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.07815},
}
```

The repository ships a `CITATION.cff` so GitHub can surface this metadata
directly on the project page.

## License

OrScale is released under the MIT License. See [`LICENSE`](LICENSE) for
details.

## Acknowledgements

OrScale builds on the orthogonalised-update line of work
([Muon](https://github.com/KellerJordan/modded-nanogpt),
[Moonlight](https://arxiv.org/abs/2502.16982)) and on classical trust-ratio
optimizers (LARS, LAMB). We thank the broader optimizer-research community for
open implementations and reproducible baselines.
