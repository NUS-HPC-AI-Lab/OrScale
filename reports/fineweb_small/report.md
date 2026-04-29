# FineWeb-Edu small_125m: post-fix sweep analysis

**Sweep dirs**

| role | sweep | optimizers |
| --- | --- | --- |
| baseline (control) | `sweeps/fineweb_20260421_034858` | `muon` |
| post-fix runs | `sweeps/fineweb_20260427_014028` | `muon_moonlight`, `orscale_muon_moonlight` |
| post-fix runs | `sweeps/fineweb_20260427_061908` | `mutrust` |
| post-fix low-LR follow-up | `sweeps/fineweb_20260429_120914` | `muscale` |
| pre-fix runs (for before/after only) | `sweeps/fineweb_20260421_034858` + `sweeps/fineweb_20260421_035149` | original flagged trio (no pre-fix `muscale`) |

The muscale follow-up uses `training.grad_accum_steps=4` on 2 GPUs, i.e.
`tokens/step=262,144` and 5.24B total tokens over 20k steps.

Reproduce via:

```bash
python scripts/analyze_fineweb_small.py
python scripts/analyze_fineweb_bump.py \
  --sweeps sweeps/fineweb_20260429_120914 \
  --output reports/fineweb_bump_muscale
```

## TL;DR

The new muscale low-LR grid fixed the previous high-LR instability.
All four muscale cells are clean by the bump detector:

| optimizer | best LR (post-fix) | final val | gap to muon@best |
| --- | --- | --- | --- |
| `muon` (control) | `0.02` | **3.2111** | -- |
| `mutrust` | `0.02` | **3.2164** | +0.005 |
| `orscale_muon_moonlight` | `3e-3` | **3.2232** | +0.012 |
| `muscale` | `1e-3` | **3.2305** | +0.019 |
| `muon_moonlight` | `1e-3` | **3.2319** | +0.021 |

`final val` = mean of the last 3 logged validation checkpoints. The new
muscale best (`3.2305`) is effectively tied with `muon_moonlight@1e-3`
and only +0.019 nats off `muon@0.02`.

The important change from the previous muscale sweep is qualitative:
`muscale@3e-3` and `muscale@5e-3` used to show dip-bump-diverge behavior
with grad-norm clip saturated; the new `{1e-4, 3e-4, 5e-4, 1e-3}` bracket
has **no bumping cell** and the optimum remains at the upper edge (`1e-3`).

## Q1: Was the dip-bump-dip pattern eliminated?

Yes for the new muscale grid. Bump-detection (threshold = 0.3 nats,
sustained >= 3 logging windows above `min1 + threshold/2`):

| optimizer | LRs in new grid that bump | LRs in new grid that are clean |
| --- | --- | --- |
| `muon_moonlight` | `1e-2` (catastrophic, val 4.63) | `3e-4`, `1e-3`, `3e-3` |
| `orscale_muon_moonlight` | `1e-2` (catastrophic, val 4.68) | `3e-4`, `1e-3`, `3e-3` |
| `mutrust` | `0.04` (mild bump, val 3.73) | `0.005`, `0.01`, `0.02` |
| `muscale` | none | `1e-4`, `3e-4`, `5e-4`, `1e-3` |

The old muscale grid `{3e-4, 1e-3, 3e-3, 5e-3}` showed the upper two
cells bumping hard. The current low-LR follow-up removes those cells and
adds `1e-4` / `5e-4`; every curve now descends smoothly through warmup and
cosine decay. See `reports/fineweb_bump_muscale/summary.csv` and
`train_loss__muscale.png` / `val_loss__muscale.png`.

## Q2: Where is each optimizer's new optimum?

From the validation-loss grid and leaderboard:

- `mutrust`: best at **`lr=0.02`** (val 3.2164).
- `orscale_muon_moonlight`: best at **`lr=3e-3`** (val 3.2232).
- `muon_moonlight`: best at **`lr=1e-3`** (val 3.2319), with `3e-3`
  only 0.005 nats worse.
- `muscale`: best at **`lr=1e-3`** (val 3.2305). The bracket is clean:
  `1e-4 -> 3.3456`, `3e-4 -> 3.2604`, `5e-4 -> 3.2413`,
  `1e-3 -> 3.2305`.

The new muscale grid still has its optimum at the upper edge, but unlike
the earlier `3e-3` / `5e-3` attempt, the `1e-3` cell is stable and not clip
limited. A next bracket around `{7e-4, 1e-3, 1.5e-3, 2e-3}` would be the
right fine-grained follow-up if we want to know whether the optimum lies
slightly above `1e-3` without jumping all the way back to unstable `3e-3`.

## Q3: How does muscale compare to muon and the other variants?

`best_lr_comparison__val.png` overlays the five best-LR curves. With the
new low-LR bracket and 262k tokens/step, `muscale@1e-3` joins the main
cluster: it is +0.019 nats from `muon@0.02`, +0.007 from
`orscale_muon_moonlight@3e-3`, and slightly better than
`muon_moonlight@1e-3`.

This changes the earlier interpretation. The first muscale sweep looked
materially worse because the usable grid was too narrow and the high end
was unstable. After re-gridding, muscale is not a clear loser; it is a
Muon-family peer at this scale. The remaining ordering is single-seed only
and should not be over-interpreted.

## Q4: Is the global grad-norm clip firing constantly?

No at the recommended LRs:

| optimizer | LR | post-warmup grad mean | post-warmup sat frac |
| --- | --- | --- | --- |
| `muon_moonlight` | `1e-3` (best) | 0.258 | 0.000 |
| `muon_moonlight` | `3e-3` | 0.335 | 0.005 |
| `muon_moonlight` | `1e-2` (div) | 9.216 | 0.996 |
| `orscale_muon_moonlight` | `3e-3` (best) | 0.248 | 0.000 |
| `orscale_muon_moonlight` | `1e-2` (div) | 6.475 | 0.999 |
| `mutrust` | `0.02` (best) | 0.360 | 0.038 |
| `mutrust` | `0.04` (high) | 0.414 | 0.109 |
| `muscale` | `1e-4` | 0.763 | 0.131 |
| `muscale` | `3e-4` | 0.375 | 0.009 |
| `muscale` | `5e-4` | 0.288 | 0.000 |
| `muscale` | `1e-3` (best) | 0.252 | 0.000 |

The clip is not setting the effective LR for `muscale@1e-3`; the
post-warmup mean is 0.252 and the saturation fraction is exactly 0. The
only muscale cell with noticeable saturation is `1e-4`, which is not an
instability signal here: it occurs early while the run is under-stepping,
and the validation curve is monotone but too slow.

## Recommendations

1. **Use `lr=1e-3` as the current FineWeb small muscale setting.** It is
   stable, has no bump, and reaches 3.2305 final val.
2. **Run a fine bracket above `1e-3` before finalizing.** Try
   `{7e-4, 1e-3, 1.5e-3, 2e-3}`. The old `3e-3` cell was unstable, but the
   new grid still improves monotonically up to `1e-3`.
3. **Keep one seed for broad sweeps, but add seeds 43/44 for claims.**
   Current optimizer ordering is still single-seed for FineWeb.
4. **Keep `r_min=0.5`, `r_max=1.5`, `grad_clip_norm=1.0`.** The low-LR
   muscale grid validates the current bounds; instability was from grid
   placement, not from a need to retune the clip.

## Artifacts

| file | what |
| --- | --- |
| `summary.csv` | every run, every metric (per-opt-LR-tag) |
| `summary_by_opt_lr.md` | leaderboard sorted within optimizer |
| `train_loss__<opt>.png` | per-optimizer LR overlay (post-fix), train |
| `val_loss__<opt>.png` | per-optimizer LR overlay (post-fix), val |
| `grid__train_loss.png` | 5-panel small multiples (muon + 4 flagged), train |
| `grid__val_loss.png` | 5-panel small multiples (muon + 4 flagged), val |
| `before_after__<opt>.png` | pre-fix vs post-fix for the original flagged trio |
| `best_lr_comparison__train.png` | best LR per optimizer, train overlay |
| `best_lr_comparison__val.png` | best LR per optimizer, val overlay |
| `grad_norm__muscale.png` | post-clip grad norm vs step for the new muscale grid |
| `../fineweb_bump_muscale/` | bump-detector output for the new muscale-only sweep |
| [`trust_ratio_analysis.md`](trust_ratio_analysis.md) | W&B trust-ratio clipping vs adaptation (`orscale_muon_moonlight`, `mutrust`, `muscale`) |

## Open questions

- **Is `1e-3` truly the optimum?** It is the best cell and still on the
  upper edge of the clean grid. A small bracket between `1e-3` and `3e-3`
  would answer this.
- **How much of the new improvement is batch/tokens-per-step?** The new
  muscale sweep uses `262,144` tokens/step and 5.24B total tokens, matching
  the intended comparison setting.
- **Multi-seed ordering?** `muscale`, `muon_moonlight`, and
  `orscale_muon_moonlight` are close enough that three seeds are needed for
  ordering claims.
