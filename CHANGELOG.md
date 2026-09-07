# Changelog

## 0.2.0 (2026-09-07)

### Added

- Moonlight-16B-A3B mixture-of-experts model (`orscale/model/moonlight_moe.py`):
  multi-head latent attention, 64 routed + 2 shared experts with top-6 routing,
  aux-loss-free router-bias balancing, activation checkpointing, and
  `moonlight_16b_a3b` / `moonlight_tiny_moe` presets. Selected with
  `model.architecture: moonlight_moe`.
- ZeRO-1-style optimizer-state sharding for Muon-family optimizers
  (`optimizer.zero_stage: 1`, `orscale/optim/zero1.py`) with per-rank
  optimizer checkpoint sidecars.
- bf16 parameter storage (`model.param_dtype`), TensorBoard logging
  (`logging.tensorboard`), and richer diagnostics: cross-layer percentiles,
  glob-selected per-parameter trust ratios, and a console summary line.
- Configs and launcher for the 10B-token Moonlight-16B-A3B optimizer
  comparison (`configs/moonlight_moe_16b_fineweb10b*.yaml`,
  `scripts/run_moonlight_moe_16b_fineweb10b.sh`), plus the headline result in
  the README.

### Changed

- `torch.compile` is applied to the Newton–Schulz kernel only when CUDA is
  available; `ORSCALE_DISABLE_TORCH_COMPILE=1` opts out.
- The `RMSNorm` fallback (PyTorch without `F.rms_norm`) normalises in fp32.

### Fixed

- `GPT.from_preset` no longer mutates the shared preset config when overrides
  are given.
- `pytest` collection under editable installs: tests that import helpers from
  `scripts/` now resolve via `[tool.pytest.ini_options].pythonpath`.

## 0.1.0 (2026-05-08)

- Initial public release.
