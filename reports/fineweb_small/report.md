# FineWeb-Edu small_125m: post-fix sweep analysis

**Sweep dirs**

| role | sweep | optimizers |
| --- | --- | --- |
| baseline (control) | `sweeps/fineweb_20260421_034858` | `muon` |
| post-fix runs | `sweeps/fineweb_20260427_014028` | `muon_moonlight`, `orscale_muon_moonlight` |
| post-fix runs | `sweeps/fineweb_20260427_061908` | `mutrust` |
| post-fix runs (added 2026-04-29) | `sweeps/fineweb_20260429_024537` | `muscale` |
| pre-fix runs (for before/after only) | `sweeps/fineweb_20260421_034858` + `sweeps/fineweb_20260421_035149` | original flagged trio (no pre-fix `muscale`) |

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

**The fix worked for the original three.** All three previously flagged
optimizers now reach a final val loss within ~0.02 nats of the `muon`
baseline at their respective new optima. The newly-swept fourth flagged
optimizer `muscale` lands +0.090 nats from `muon` (~0.07 nats further
off than the next-worst optimizer at its own optimum) and inherits the
same Moonlight-style LR-grid pathology:

| optimizer | best LR (post-fix) | final val | gap to muon@best |
| --- | --- | --- | --- |
| `muon` (control)        | `0.02`  | **3.2111** | -- |
| `mutrust`               | `0.02`  | **3.2164** | +0.005 |
| `orscale_muon_moonlight`| `3e-3`  | **3.2232** | +0.012 |
| `muon_moonlight`        | `1e-3`  | **3.2319** | +0.021 |
| `muscale`               | `1e-3`  | **3.3015** | +0.090 |

`final val` = mean of last 3 logged val checkpoints
(steps `19500/20000/20000`).

The dip-bump-dip pattern is **eliminated** at the new optima. Four LR cells
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
- `muscale@3e-3` and `muscale@5e-3` both bump catastrophically
  (val 4.49 / 4.79; `dip->BUMP->dip` confirmed by
  `analyze_fineweb_bump.py`, see
  `reports/fineweb_bump_muscale/`). The post-warmup grad-norm saturates
  the 1.0 clip on **100%** / **98.5%** of steps respectively. The
  optimum sits at `lr=1e-3` -- one step below the upper edge of the
  initial muscale grid -- so the *useful* part of this grid is just
  `{3e-4, 1e-3}`. The next sweep should drop `{3e-3, 5e-3}` and add
  `{1e-4, 5e-4}` (and possibly a second seed at `1e-3`) to bracket the
  optimum tightly and confirm whether the +0.07-nat gap to the
  Moonlight pair is real or a single-seed accident.

## Q1: Was the dip-bump-dip pattern eliminated?

Yes for every recommended cell. Bump-detection (threshold = 0.3 nats,
sustained >= 3 logging windows above `min1 + threshold/2`):

| optimizer | LRs in new grid that bump | LRs in new grid that are clean |
| --- | --- | --- |
| `muon_moonlight`         | `1e-2` (catastrophic, val 4.63) | `3e-4`, `1e-3`, `3e-3` |
| `orscale_muon_moonlight` | `1e-2` (catastrophic, val 4.68) | `3e-4`, `1e-3`, `3e-3` |
| `mutrust`                | `0.04` (mild bump, val 3.73)    | `0.005`, `0.01`, `0.02` |
| `muscale`                | `3e-3` (val 4.49, +1.53 / -1.31), `5e-3` (val 4.79, +0.85 / -0.55) | `3e-4`, `1e-3` |

Compare to the pre-fix sweep, where **every original-flagged-optimizer
/LR combination bumped except `orscale_muon_moonlight@0.005`** (10/12
original-flagged runs bumped pre-fix; 5/16 bump post-fix across the four
flagged optimizers, of which only 1 still converges within 0.6 nats of
the cluster -- `mutrust@0.04`).

`muscale` did not have a pre-fix sweep, so the before/after plot is
omitted for it; instead see `train_loss__muscale.png` /
`val_loss__muscale.png` for the four LRs side-by-side. The story is
identical to the other Moonlight-scaled optimizers: the two upper LRs
(`3e-3`, `5e-3`) blow through min1 around step ~1500-2000 and never
recover (the `5e-3` curve actually starts climbing during warmup at
step ~1100), while the two lower LRs (`3e-4`, `1e-3`) glide
monotonically to the 3.30-3.33 plateau. The bump severity at
`muscale@3e-3` (+1.53 nats up / -1.31 down) is comparable to what
`muon_moonlight@1e-2` shows post-fix, suggesting the muscale grid is
~3x too high at its top edge.

See `before_after__<opt>.png` for the visual diff on the original
flagged trio. For `muon_moonlight` for example, every dashed (old)
curve crosses 5+ nats during steps 1000-3000; every solid (new) curve
except `lr=1e-2` glides monotonically toward 3.23.

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
- `muscale`                : peaks at **`lr=1e-3`** (val 3.3015). The
  grid is `{3e-4, 1e-3, 3e-3, 5e-3}` and the upper half blows up; the
  monotone half is `{3e-4, 1e-3}` with a 0.026-nat gap between them.
  We do not yet know whether the optimum is bracketed below: a cell at
  `5e-4` (and possibly `1e-4`) would be informative.

Notably, `mutrust`'s optimum lining up with `muon`'s makes sense: with
`r_max=1.5` the trust ratio caps the per-layer step at ~1.5 *
shape_scale, which is roughly Moonlight's per-step magnitude with a
~1.5x extra margin. The MUON grid at `lr=0.02` is then the right LR.

`muscale`'s optimum lining up with `muon_moonlight`'s also makes
sense: both apply the Moonlight `0.2*sqrt(max(m,n))` shape rescaling,
so the per-layer update is in the same magnitude regime; the only
difference is that `muscale` uses an `RMS(M_hat)` denominator instead
of `||·||_F` in the trust ratio. Empirically that change pushes
`muscale` ~0.07 nats worse than `muon_moonlight` at the same LR -- a
larger gap than any of the other three flagged optimizers ends up at,
and worth investigating before recommending `muscale` as a Muon
replacement at this scale.

## Q3: How does each post-fix optimizer compare to muon at its best?

`best_lr_comparison__val.png` overlays the five runs (muon + four
flagged) at their best LRs. The gap from `muon@best` to the next three
is **0.005 - 0.021 nats** -- i.e. `mutrust`, `orscale_muon_moonlight`
and `muon_moonlight` all converge to essentially the same loss level
as `muon`. `muscale` opens up a noticeably larger gap of **+0.090
nats** (3.3015 vs 3.2111) at its best LR, and that gap is *not*
closing toward the end of training (the curve plateaus at 3.30 from
step ~17000 onwards). At single-seed resolution this is well above
the per-optimizer ordering noise we see for the other three, but it
is plausibly within seed variance for the model; a 2-3 seed re-run
is warranted before declaring `muscale` strictly worse here (see
recommendations).

None of the post-fix flagged optimizers beats `muon` outright on this
20k-step / 5.2B-token target; the most likely interpretation is that
for FineWeb-Edu small_125m, the basic Muon update is already well-
tuned and the trust ratio / shape rescaling tricks deliver no net
win at this scale (they do match it within noise for three of four,
however, which is the design goal). `muscale`'s underperformance --
specifically the change to an RMS(M_hat) denominator -- seems to cost
this scale ~0.07 nats relative to a ||·||_F denominator at otherwise
identical settings.

The **largest pre/post-fix gain** is `muon_moonlight`, which went from
final val 4.32 -> 3.23 (~1.1 nats lower) just by moving onto the
correct LR grid; `mutrust` improved 4.44 -> 3.22 (~1.2 nats); the wins
came almost entirely from fixing the LR scale, with the `r_max` tighten
and grad-clip catching the residual instabilities. `muscale` has no
pre-fix run to compare against, but its post-fix sweep starts from the
already-known-good Moonlight grid, so the equivalent "regime fix" was
applied implicitly from the start.

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
| `muscale` | `3e-4`        | 0.413 | 0.010 |
| `muscale` | `1e-3` (best) | 0.294 | 0.000 |
| `muscale` | `3e-3` (div)  | 7.119 | 1.000 |
| `muscale` | `5e-3` (div)  | 13.267 | 0.985 |

The 1.0 clip behaves as a safety net: at the optima it costs us nothing
(grad norms sit at ~0.25-0.4), but it engages aggressively the moment a
run starts to diverge. The two original `1e-2` runs and the new
`muscale@{3e-3, 5e-3}` runs are all pinned at the 1.0 boundary for the
entire post-warmup -- meaning the *effective* optimizer step is driven
by the clip rather than by the schedule, and the optimizer never
recovers (loss stays in the 4.5-4.8 region). This indicates the clip
is set correctly: tight enough to prevent NaN-style blowups, loose
enough not to distort the well-tuned LRs.

`grad_norm__muscale.png` mirrors the
`grad_norm__{muon_moonlight,orscale_muon_moonlight}.png` pattern very
closely: the two converging LRs sit a hair below `0.5` for the
entire post-warmup, while the two diverging LRs ride at the `1.0`
boundary continuously (with multi-second-order excursions when the
loss spikes -- `muscale@5e-3` shows an isolated grad-norm spike to
~50 around step 1500, where the loss starts climbing through the
warmup-end `min1`). The familiar story: once the underlying problem
is too aggressive a step magnitude, capping the gradient norm only
caps the visible blow-up, it does not save the optimizer.

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
3. **Re-grid `muscale` to `{1e-4, 3e-4, 5e-4, 1e-3}`** for the next
   sweep. The current `{3e-4, 1e-3, 3e-3, 5e-3}` grid has `3e-3` and
   `5e-3` both diverging (grad-norm clip at 100%) and the converging
   half is `{3e-4, 1e-3}` with `1e-3` slightly better; a denser grid
   below `1e-3` is required to (a) verify the optimum isn't actually
   at `5e-4`, and (b) close the +0.07-nat gap to the other Moonlight
   variants. Keep `r_min=0.5`, `r_max=1.5`, `grad_clip_norm=1.0`
   unchanged -- they behave identically to the other flagged
   optimizers at the converging LRs.
4. **Add seeds 43, 44 at the per-optimizer winner** (`muon@0.02`,
   `mutrust@0.02`, `orscale_muon_moonlight@3e-3`,
   `muon_moonlight@1e-3`, `muscale@1e-3`) to confirm the
   0.005-0.090 nat gaps to `muon` are real and not seed noise -- the
   first three gaps are within typical seed-to-seed variation for
   this model size, but `muscale`'s 0.090-nat gap is at the edge of
   what 1 seed can justify and worth nailing down with 3 seeds before
   making any optimizer recommendation.
5. **No further changes** to `r_min`/`r_max` are warranted: the
   `clip_active_active_frac` curves (in W&B) show the trust ratio
   smoothly de-saturating after warmup at the recommended LRs, and the
   tighter bound is what kept `orscale_muon_moonlight` from blowing up
   like it did pre-fix at `lr=0.01`. The same holds for `muscale`
   at `lr<=1e-3` -- the trust ratio is well inside `[0.5, 1.5]` post-
   warmup (visible in the W&B `trust_ratio_*_mean` panels for
   `small_125m-muscale-lr0.001-seed42`).

## Artifacts

| file | what |
| --- | --- |
| `summary.csv`                   | every run, every metric (per-opt-LR-tag) |
| `summary_by_opt_lr.md`          | leaderboard sorted within optimizer |
| `train_loss__<opt>.png`         | per-optimizer LR overlay (post-fix), train (one per optimizer including `muscale`) |
| `val_loss__<opt>.png`           | per-optimizer LR overlay (post-fix), val (one per optimizer including `muscale`) |
| `grid__train_loss.png`          | 5-panel small multiples (muon + 4 flagged), train |
| `grid__val_loss.png`            | 5-panel small multiples (muon + 4 flagged), val |
| `before_after__<opt>.png`       | pre-fix (dashed grey) vs post-fix (color), train; only emitted for the original flagged trio (no pre-fix `muscale`) |
| `best_lr_comparison__train.png` | best LR per opt, overlay, train (5 lines) |
| `best_lr_comparison__val.png`   | best LR per opt, overlay, val (5 lines) |
| `grad_norm__<opt>.png`          | post-clip grad norm vs step (per opt; includes `grad_norm__muscale.png`) |
| `../fineweb_bump_muscale/`      | bump-detector output for the muscale-only sweep (CSV + per-LR overlays) |

## Open questions

- **Multi-seed?** Current sweep is one seed per cell. The 0.005-0.090
  nats spread between the five optimizers is at the edge of what one
  seed can resolve for the top three; for `muscale` the 0.07-nat gap
  to the rest is large enough that a 3-seed re-run at `lr=1e-3` is
  the single highest-value follow-up.
- **Drop `lr=1e-2` and add `5e-3`?** Or keep `1e-2` as a tripwire so
  future regressions in the Moonlight scale show up immediately?
- **`muscale` LR bracket below `1e-3`?** Per Q2 we don't know if
  `1e-3` is the optimum or a saddle on the way down. Add `5e-4` (and
  optionally `1e-4`) before claiming the optimum is bracketed.
- **Why does `muscale` cost ~0.07 nats vs `muon_moonlight`?** Both
  apply Moonlight shape-rescaling and the same trust-ratio clip; the
  only difference is the trust-ratio denominator (`RMS(M_hat)` vs
  `||·||_F`). At small_125m / 5.2B tokens the RMS choice consistently
  trails. Worth checking whether the gap is constant in steps (and
  hence likely to vanish at longer horizons) or constant in nats (a
  fundamental floor) by extending the best `muscale` run to 40k steps.
- **Larger model?** All conclusions above are at small_125m / 5.2B
  tokens. The expected story at `pilot_25m` is "no problem at all";
  at `medium_350m` (more steps, larger layers) the trust-ratio
  saturation pattern at the upper LRs may shift -- worth a re-check
  before promoting these defaults to medium.
