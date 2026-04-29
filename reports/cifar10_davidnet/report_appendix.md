## MuScale follow-up sweep (2026-04-29)

`muscale` is a fresh OrScale variant added on 2026-04-22 (Nesterov
momentum, `RMS(M_hat)` denominator, Moonlight `0.2*sqrt(max(m,n))`
shape rescaling, no partial exponent -- i.e. it is `muscale_alpha`
with `alpha=1.0`). It joined the CIFAR-10 / DavidNet sweep on
2026-04-29 with a 4-LR x 3-seed grid that mirrors the original Muon /
OrScale grid:

- **Sweep dir**: `sweeps/20260429_023319/` (12 runs, all completed)
- **LR grid**:   `{0.005, 0.01, 0.02, 0.04}` -- same as Muon /
  OrScale-Muon / OrScale-Muon-WD / OrScale-Muon-Moonlight / mutrust
- **Seeds**:     `42, 43, 44`
- **Settings**:  config defaults from `configs/cifar10_davidnet.yaml`
                 (`r_min=0.5`, `r_max=1.5`, `momentum=0.95`,
                 `weight_decay=5e-4`, `eps=1e-6`, `ns_iters=5`)

### Headline result

| LR    | val_top1 (mean ± std, last 3 epochs) | wallclock (s)  |
| ----- | ------------------------------------ | -------------- |
| 0.005 | 93.61 ± 0.16                         | 94.5  ± 3.3    |
| 0.01  | 93.65 ± 0.07                         | 112.3 ± 18.2   |
| 0.02  | 93.56 ± 0.07                         | 123.0 ± 0.6    |
| 0.04  | **93.75 ± 0.08**                     | 101.1 ± 19.3   |

The best LR is `0.04` at the upper edge of the swept grid -- the same
LR Muon prefers, and one click higher than the OrScale-Moonlight
optimum at `0.02`. The val_top1 surface is essentially flat across
the four LRs (93.56 -> 93.75 = 0.19 nats span; std on each cell is
0.07-0.16), so the optimum is broad and seed noise dominates the
LR-sensitivity ordering at the tighter end.

### Where it lands in the head-to-head

In the full 9-optimizer ranking (see `## Ranking at best LR` above),
`muscale@0.04` ties `muon_moonlight@0.01` at 93.75% (rank 5/6 by
mean) and slightly beats vanilla `muon@0.04` (93.70%, rank 7). It
sits **0.30 nats below** the three-way tie at the top
(`orscale_muon`, `orscale_muon_wd`, `orscale_muon_moonlight`, all
at 94.05-94.06% at lr=0.02) and **0.09 nats below** `mutrust@0.005`
(rank 4, 93.84%).

Two takeaways for muscale at this scale (DavidNet / CIFAR-10, 24
epochs, ~6.5M params, batch 512):

1. **No instability anywhere on the grid.** Unlike at the
   FineWeb-Edu / 125M-param scale (see `reports/fineweb_small/`),
   the trust-ratio / RMS-denominator combo is *not* exploding at
   the upper LRs here. The 24-epoch cosine + warmup schedule and
   the smaller per-layer dimensions both push the per-step update
   into a benign regime, and the `r_max=1.5` clip keeps it there.
2. **Performance is flat-and-slightly-low.** Muscale's RMS-based
   trust ratio doesn't deliver the +0.3-nat gain that
   OrScale-Moonlight gets from its `||·||_F`-based ratio at this
   scale. The four converging cells cluster within 0.19 nats and
   never reach the 94% bar that the OrScale trio clears at lr=0.02.
   This is consistent with the FineWeb-Edu finding that
   `RMS(M_hat)` is ~0.07 nats worse than `||·||_F` at the LM scale
   too, so the gap is reproducible across modalities.

### Recommendations

1. **Try a higher LR.** The best cell `0.04` is on the boundary
   and the surface is flat; try `{0.04, 0.06, 0.08, 0.12}` for a
   targeted follow-up on muscale only.
2. **Don't promote muscale as the CIFAR default** until point 1
   resolves whether the optimum is actually at `0.04` or higher.
   Even at the upper bound, the gap to OrScale-Moonlight (-0.30
   nats) is well outside seed noise (std ≈ 0.10), so muscale is
   clearly behind on this benchmark in its current form.
3. **Keep `r_min=0.5`, `r_max=1.5`.** No CIFAR run shows the
   trust ratio pinned at the upper bound, and dropping the bounds
   further would only matter if something started diverging --
   nothing did here.

### Artifacts

| file | what |
| --- | --- |
| `sweeps/20260429_023319/sweep_config.json` | exact run grid |
| `sweeps/20260429_023319/name=muscale_lr=*_seed=*.log` | per-run stdout |
| `runs.csv`, `epochs.csv`, `steps.csv` | tidy data (muscale rows included) |
| `summary_by_opt_lr.md` / `.csv` | leaderboard with muscale block |
| `plots/curves_val_top1.png` | best-LR head-to-head (muscale = olive) |
| `plots/lr_sensitivity_val_top1.png` | LR sweep including muscale |
| `plots/variance_best_lr_val_top1.png` | per-seed bars at each best LR |
| `plots/wallclock_seconds.png` | wall-clock comparison |
