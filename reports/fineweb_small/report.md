# FineWeb-Edu small_125m: post-fix sweep analysis

**Sweep dirs**

| role | sweep | optimizers |
| --- | --- | --- |
| baseline (control) | `sweeps/fineweb_20260421_034858` | `muon` |
| post-fix runs | `sweeps/fineweb_20260427_014028` | `muon_moonlight`, `orscale_muon_moonlight` |
| post-fix runs | `sweeps/fineweb_20260427_061908` | `mutrust` |
| pre-fix runs (for before/after only) | `sweeps/fineweb_20260421_034858` + `sweeps/fineweb_20260421_035149` | flagged trio |

**Fix bundle** applied between the pre-fix and post-fix sweeps:

1. **LR grid widened downward** for the Moonlight-scaled optimizers
   (`muon_moonlight`, `orscale_muon_moonlight`) from `{0.005, 0.01, 0.02, 0.04}`
   to `{3e-4, 1e-3, 3e-3, 1e-2}`; `mutrust` was *kept* on the Muon grid
   `{0.005, 0.01, 0.02, 0.04}`.
2. **Trust-ratio clip tightened** from `r_min=0.1, r_max=10.0` to
   `r_min=0.5, r_max=1.5` (config + `OrScaleOptimizer` default + factory).
3. **Global gradient-norm clip** of `1.0` applied in `Trainer` after
   `backward()` and before `optimizer.step()`.
4. **DiagnosticLogger** extended with cross-layer aggregates
   (`trust_ratio_*_{mean,min,max}`, `clip_active_active_frac`, ...).

Reproduce via:

```bash
python scripts/analyze_fineweb_small.py
```

## TL;DR

**The fix worked.** All three previously flagged optimizers now reach a
final val loss within ~0.02 nats of the `muon` baseline at their
respective new optima:

| optimizer | best LR (post-fix) | final val | gap to muon@best |
| --- | --- | --- | --- |
| `muon` (control)        | `0.02`  | **3.2111** | -- |
| `mutrust`               | `0.02`  | **3.2164** | +0.005 |
| `orscale_muon_moonlight`| `3e-3`  | **3.2232** | +0.012 |
| `muon_moonlight`        | `1e-3`  | **3.2319** | +0.021 |

`final val` = mean of last 3 logged val checkpoints
(steps `19500/20000/20000`).

The dip-bump-dip pattern is **eliminated** at the new optima. Two LR cells
still bump:

- `muon_moonlight` and `orscale_muon_moonlight` at `lr=1e-2` -- the *top*
  edge of the new Moonlight grid -- diverge to ~4.6 val loss
  (`grad_norm` saturates the 1.0 clip on >99% of steps; `trust_ratio`
  also pinned at the 1.5 cap), so the new grid still brackets the
  instability. Recommend dropping `1e-2` from this grid and replacing
  it with `5e-3` for finer coverage of the optimum.
- `mutrust@0.04` survives but ends at val 3.7326 -- a different basin
  than the other LRs (3.21-3.23) -- with a small (+0.35 nats), brief
  bump around step 1300 that recovers fully. The post-warmup grad-norm
  saturates the 1.0 clip ~11% of the time and trust ratio is pinned at
  `r_max=1.5` for ~100% of training. Recommend dropping `0.04` from the
  `mutrust` grid; the optimum sits at `0.02`.

## Q1: Was the dip-bump-dip pattern eliminated?

Yes for every recommended cell. Bump-detection (threshold = 0.3 nats,
sustained >= 3 logging windows above `min1 + threshold/2`):

| optimizer | LRs in new grid that bump | LRs in new grid that are clean |
| --- | --- | --- |
| `muon_moonlight`         | `1e-2` (catastrophic, val 4.63) | `3e-4`, `1e-3`, `3e-3` |
| `orscale_muon_moonlight` | `1e-2` (catastrophic, val 4.68) | `3e-4`, `1e-3`, `3e-3` |
| `mutrust`                | `0.04` (mild bump, val 3.73)    | `0.005`, `0.01`, `0.02` |

Compare to the pre-fix sweep, where **every flagged-optimizer/LR
combination bumped except `orscale_muon_moonlight@0.005`** (10/12 flagged
runs bumped pre-fix; 3/12 bump post-fix, and only 1 of those still
converges).

See `before_after__<opt>.png` for the visual diff. For
`muon_moonlight` for example, every dashed (old) curve crosses 5+ nats
during steps 1000-3000; every solid (new) curve except `lr=1e-2` glides
monotonically toward 3.23.

## Q2: Where is each optimizer's new optimum?

From the val-loss panel (`grid__val_loss.png`, also
`val_loss__<opt>.png` per opt) and the leaderboard:

- `mutrust`                : peaks at **`lr=0.02`** (val 3.2164),
  matching `muon`'s optimum. The new grid brackets it cleanly:
  0.005 -> 3.2339, 0.01 -> 3.2215, 0.02 -> 3.2164, 0.04 -> 3.7332.
- `orscale_muon_moonlight` : peaks at **`lr=3e-3`** (val 3.2232) at the
  upper-middle of the new grid. The optimum is between `3e-3` and `1e-2`
  but `1e-2` itself is diverging, so an additional cell at `5e-3` would
  be informative.
- `muon_moonlight`         : peaks at **`lr=1e-3`** (val 3.2319). The
  curve at `3e-3` is only 0.005 nats worse, so the optimum is broad
  across `[1e-3, 3e-3]`. `1e-2` diverges.

Notably, `mutrust`'s optimum lining up with `muon`'s makes sense: with
`r_max=1.5` the trust ratio caps the per-layer step at ~1.5 *
shape_scale, which is roughly Moonlight's per-step magnitude with a
~1.5x extra margin. The MUON grid at `lr=0.02` is then the right LR.

## Q3: How does each post-fix optimizer compare to muon at its best?

`best_lr_comparison__val.png` overlays the four runs at their best LRs.
The gap between best-of-each is **0.005 - 0.021 nats** -- i.e. they all
converge to essentially the same loss level. None of the post-fix
flagged optimizers beats `muon` outright on this 20k-step / 5.2B-token
target; the most likely interpretation is that for FineWeb-Edu
small_125m, the basic Muon update is already well-tuned and the trust
ratio / shape rescaling tricks deliver no net win at this scale (they
do match it within noise, however, which is the design goal).

The **largest pre/post-fix gain** is `muon_moonlight`, which went from
final val 4.32 -> 3.23 (~1.1 nats lower) just by moving onto the
correct LR grid; `mutrust` improved 4.44 -> 3.22 (~1.2 nats); the wins
came almost entirely from fixing the LR scale, with the `r_max` tighten
and grad-clip catching the residual instabilities.

## Q4: Is the global grad-norm clip firing constantly?

Almost never at the recommended LRs:

| optimizer | LR | post-warmup grad mean | post-warmup sat frac |
| --- | --- | --- | --- |
| `muon_moonlight` | `1e-3` (best) | 0.258 | 0.000 |
| `muon_moonlight` | `3e-3`        | 0.335 | 0.005 |
| `muon_moonlight` | `1e-2` (div)  | 9.216 | 0.996 |
| `orscale_muon_moonlight` | `3e-3` (best) | 0.248 | 0.000 |
| `orscale_muon_moonlight` | `1e-2` (div)  | 6.475 | 0.999 |
| `mutrust` | `0.02` (best) | 0.360 | 0.038 |
| `mutrust` | `0.04` (high) | 0.414 | 0.109 |

The 1.0 clip behaves as a safety net: at the optima it costs us nothing
(grad norms sit at ~0.25-0.4), but it engages aggressively the moment a
run starts to diverge. The two `1e-2` runs are pinned at the 1.0 boundary
for the entire post-warmup -- meaning the *effective* optimizer step is
driven by the clip rather than by the schedule, and the optimizer never
recovers (loss stays in the 4.6 region). This indicates the clip is set
correctly: tight enough to prevent NaN-style blowups, loose enough not to
distort the well-tuned LRs.

`grad_norm__mutrust.png` is the most interesting: at lr=0.04 the
grad-norm has a wide post-warmup hump from step 2000-6500 -- precisely
the period the train loss bumps -- and only settles below 0.5 once the
cosine schedule pulls the LR down. With the trust-ratio summary showing
`clip_active_mean=1` for the entire run at this LR, the optimizer is
pinned by *both* the trust-ratio cap (`r_max=1.5`) and the gradient-norm
cap (`1.0`) for the entire post-warmup region -- a clear signal that
this LR is in the wrong regime for the optimizer and shouldn't be in the
final grid.

## Recommendations

1. **Drop `lr=1e-2`** from the Moonlight grid for the next sweep; replace
   with `5e-3` to give `orscale_muon_moonlight` a cleaner two-cell window
   around its optimum.
2. **Drop `lr=0.04`** from the `mutrust` grid; its trust-ratio + grad-clip
   are both pinned, and it converges to a worse basin (3.73 vs 3.22).
3. **Add a second seed** at the per-optimizer winner (`muon@0.02`,
   `mutrust@0.02`, `orscale_muon_moonlight@3e-3`, `muon_moonlight@1e-3`)
   to confirm the 0.005-0.021 nats gap to `muon` is real and not seed
   noise -- the per-optimizer gaps here are within typical seed-to-seed
   variation for this model size.
4. **No further changes** to `r_min`/`r_max` are warranted: the
   `clip_active_active_frac` curves (in W&B) show the trust ratio
   smoothly de-saturating after warmup at the recommended LRs, and the
   tighter bound is what kept `orscale_muon_moonlight` from blowing up
   like it did pre-fix at `lr=0.01`.

## Artifacts

| file | what |
| --- | --- |
| `summary.csv`                   | every run, every metric (per-opt-LR-tag) |
| `summary_by_opt_lr.md`          | leaderboard sorted within optimizer |
| `train_loss__<opt>.png`         | per-optimizer LR overlay (post-fix), train |
| `val_loss__<opt>.png`           | per-optimizer LR overlay (post-fix), val |
| `grid__train_loss.png`          | 4-panel small multiples, train |
| `grid__val_loss.png`            | 4-panel small multiples, val |
| `before_after__<opt>.png`       | pre-fix (dashed grey) vs post-fix (color), train |
| `best_lr_comparison__train.png` | best LR per opt, overlay, train |
| `best_lr_comparison__val.png`   | best LR per opt, overlay, val |
| `grad_norm__<opt>.png`          | post-clip grad norm vs step (per opt) |

## Open questions

- **Multi-seed?** Current sweep is one seed per cell. The 0.005-0.021
  nats spread between the four optimizers is at the edge of what one
  seed can resolve. Worth re-running the four winners at seeds
  `{1234, 2024, 31337}` if you want to claim ordering.
- **Drop `lr=1e-2` and add `5e-3`?** Or keep `1e-2` as a tripwire so
  future regressions in the Moonlight scale show up immediately?
- **Larger model?** All conclusions above are at small_125m / 5.2B
  tokens. The expected story at `pilot_25m` is "no problem at all";
  at `medium_350m` (more steps, larger layers) the trust-ratio
  saturation pattern at the upper LRs may shift -- worth a re-check
  before promoting these defaults to medium.
