# FineWeb small_125m: trust-ratio diagnostics

This report answers whether **trust ratios materially adapt** for
`orscale_muon_moonlight`, `mutrust`, and `muscale` on FineWeb-Edu small_125m,
or whether they sit at **`r_min`** / **`r_max`** most of the time.

**Evidence source:** local W&B run files under `wandb/run-*-<id>/run-<id>.wandb`,
decoded with the same W&B SDK as training (not only `wandb-summary.json`, which
is mostly a last-step snapshot). Each completed run has **400** diagnostic rows
(`log_every=50`, `max_steps=20000`). Config uses **`r_min=0.5`**, **`r_max=1.5`**
(post-fix sweeps).

**Reproduce:**

```bash
conda run -n orscale python scripts/analyze_fineweb_trust_ratio.py
```

(Run from repo root; the script drops the repo root from `sys.path` so
`import wandb` resolves to the SDK, not the `wandb/` output directory.)

---

## Verdict (per variant)

| Variant | Verdict | Notes |
| --- | --- | --- |
| **`mutrust`** | **Not adaptive** (upper-saturated) | Cross-layer mean clipped trust is at **`r_max=1.5`** for **99.75–100%** of diagnostic steps at every LR in the post-fix grid. Raw ratio means are **≫ 1.5** (hundreds to tens of thousands). |
| **`muscale`** | **Not adaptive** (upper-saturated) | Same pattern: **99.5–100%** of steps have mean clipped at **`r_max`**. Raw means stay **≫ `r_max`**; instability at high LR is consistent with trust doing almost nothing until clip. |
| **`orscale_muon_moonlight`** | **Mixed / partially adaptive** | At **`lr=3e-3`** (best val in the post-fix sweep), mean clipped trust **varies** (min 0.5, max ~0.74); only **15.5%** of steps have mean exactly at **`r_min`**, **0%** stuck at **`r_max`** — trust **does** take effect. At **`lr=1e-3`** and **`3e-4`**, mean clipped trust is **exactly `r_min`** for **100%** of steps (fully lower-saturated). At **`lr=1e-2`**, **84%** of steps have mean at **`r_max`** (mostly upper-saturated). |

---

## Evidence table (W&B history aggregates)

Columns:

- **n_diag:** number of logged diagnostic steps (400 for full runs).
- **frac_clip_eq1:** fraction of steps where `diagnostics/_summary/clip_active_mean == 1` (all layers clipping).
- **clip_mean_*:** min / mean / max of `diagnostics/_summary/trust_ratio_clipped_mean` over time.
- **frac@r_min / frac@r_max:** fraction of steps where mean clipped trust equals **0.5** / **1.5** (numerical tolerance 1e-9).
- **raw_mean_*:** min / mean / max of `diagnostics/_summary/trust_ratio_raw_mean` over time.

| tag | opt | lr | run_id | n | frac_clip_eq1 | clip_mean (min–mean–max) | frac@r_min | frac@r_max | raw_mean (min–mean–max) | final val |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| postfix | orscale_muon_moonlight | 1e-2 | kjmcykks | 400 | 0.88 | 0.50 – 1.43 – 1.50 | 0.04 | 0.84 | 0.07 – 3.84 – 4.86 | 4.678 |
| postfix | orscale_muon_moonlight | 1e-3 | akns63sa | 400 | 1.00 | 0.50 – 0.50 – 0.50 | 1.00 | 0.00 | 0.07 – 0.20 – 0.24 | 3.255 |
| postfix | orscale_muon_moonlight | 3e-3 | cnlqrsck | 400 | 0.16 | 0.50 – 0.64 – 0.74 | 0.16 | 0.00 | 0.07 – 0.59 – 0.73 | **3.223** |
| postfix | orscale_muon_moonlight | 3e-4 | esdi2j4r | 400 | 1.00 | 0.50 – 0.50 – 0.50 | 1.00 | 0.00 | 0.06 – 0.09 – 0.13 | 3.344 |
| postfix | mutrust | 0.005 | 7bn5voyd | 400 | 1.00 | 1.42 – 1.50 – 1.50 | 0.00 | 1.00 | 24.5 – 1222 – 1590 | 3.234 |
| postfix | mutrust | 0.01 | cgqhcatj | 400 | 1.00 | 1.49 – 1.50 – 1.50 | 0.00 | 1.00 | 35.2 – 3332 – 4481 | 3.221 |
| postfix | mutrust | 0.02 | ivtad6sy | 400 | 1.00 | 1.50 – 1.50 – 1.50 | 0.00 | 1.00 | 74.8 – 7492 – 10494 | **3.216** |
| postfix | mutrust | 0.04 | 9x1gdn83 | 400 | 1.00 | 1.50 – 1.50 – 1.50 | 0.00 | 1.00 | 130 – 19128 – 67416 | 3.733 |
| muscale_bump | muscale | 1e-3 | qfczrs3x | 400 | 1.00 | 1.49 – 1.50 – 1.50 | 0.00 | 1.00 | 34.2 – 3196 – 4151 | 3.301 |
| muscale_bump | muscale | 3e-3 | lawgc9de | 400 | 1.00 | 1.50 – 1.50 – 1.50 | 0.00 | 1.00 | 102 – 26983 – 58290 | 4.491 |
| muscale_bump | muscale | 3e-4 | 82ynpt4t | 400 | 0.99 | 1.40 – 1.50 – 1.50 | 0.00 | 0.99 | 23.8 – 446 – 549 | 3.327 |
| muscale_bump | muscale | 5e-3 | wlu4sq4p | 400 | 1.00 | 1.50 – 1.50 – 1.50 | 0.00 | 1.00 | 143 – 75387 – 116844 | 4.788 |
| postfix | muscale | 1e-3 | ugwiewfe | 400 | 1.00 | 1.49 – 1.50 – 1.50 | 0.00 | 1.00 | 34.0 – 3874 – 5146 | **3.230** |
| postfix | muscale | 1e-4 | 3sx0n05g | 400 | 0.99 | 1.38 – 1.50 – 1.50 | 0.00 | 0.99 | 21.6 – 150 – 433 | 3.345 |
| postfix | muscale | 3e-4 | 40g0tpp2 | 400 | 0.99 | 1.39 – 1.50 – 1.50 | 0.00 | 0.99 | 23.7 – 542 – 680 | 3.260 |
| postfix | muscale | 5e-4 | 81jofovf | 400 | 1.00 | 1.42 – 1.50 – 1.50 | 0.00 | 1.00 | 24.5 – 1228 – 1600 | 3.241 |

Sweep mapping: `sweeps/fineweb_20260427_014028` (orscale_muon_moonlight),
`sweeps/fineweb_20260427_061908` (mutrust), `sweeps/fineweb_20260429_024537`
(muscale bump), `sweeps/fineweb_20260429_120914` (muscale post-fix).

---

## Why this happens (mechanism)

All three variants live in `OrScaleOptimizer` (`orscale/optim/orscale_optimizer.py`):

- **`mutrust`:** \(r = \|W\|_F / (\|\hat M\|_F + \varepsilon)\) with Nesterov \(\hat M\).
  During training, **momentum buffers stay much smaller in Frobenius norm than
  weights**, so the ratio is **huge** and clips to **`r_max`** almost always.

- **`muscale`:** \(r = \mathrm{RMS}(W) / (\mathrm{RMS}(\hat M) + \varepsilon)\).
  Same story: **RMS(\(\hat M\))** is tiny relative to **RMS(\(W\))** on these
  runs, so raw \(r \gg r_{\max}\) and the effective step uses **`r_max`** almost
  everywhere. Moonlight **`shape_scale`** scales the orthogonal update but does
  **not** enter this ratio.

- **`orscale_muon_moonlight`:** denominator is \(\|\lambda W + \texttt{shape\_scale}\cdot Q\|_F\),
  which **couples weight scale and the orthogonal direction**. That can land
  **inside** \([r_{\min}, r_{\max}]\) for some LRs (notably **3e-3** here), or
  below **`r_min`** when the denominator is large (low LR), or above **`r_max`**
  when the effective “gradient matrix” norm is small (high LR).

So: **mutrust** and **muscale** trust ratios are **mostly a fixed multiplier
`r_max`** under the current scale of \(W\) vs \(\hat M\), while **orscale_muon_moonlight**
can still behave like a **state-dependent** gain when the denominator tracks the
update path.

Diagnostics are aggregated in `orscale/diagnostics/logger.py` as
`diagnostics/_summary/trust_ratio_*` and `clip_active_*`; per-layer keys exist
under `diagnostics/<param>/...`.

---

## How to improve

1. **Recalibrate the ratio for mutrust / muscale** so typical raw \(r\) lies in
   \([r_{\min}, r_{\max}]\) without always clipping — e.g. use a denominator
   closer in scale to the numerator (closer to the Moonlight / coupled form),
   or a **temperature** \(r = (\text{ratio})^\beta\) with \(\beta < 1\) (the
   code already has **`muscale_alpha`** for fractional power on muscale-style
   RMS stats).

2. **Schedule-aware or running bounds:** widen or shift **`r_min`/`r_max`** during
   warmup, or set bounds from **percentiles** of raw \(r\) over a window so
   clipping is rare by construction.

3. **Richer logging:** add **`clip_upper_active_frac`** / **`clip_lower_active_frac`**
   (fraction of layers with raw \(> r_{\max}\) vs raw \(< r_{\min}\)) so
   `clip_active_mean` is not the only headline; optionally log median trust
   ratios to reduce sensitivity to a few huge layers.

4. **Validation sweeps:** (a) grid **`muscale_alpha`** \(\alpha \in \{0.25, 0.5, 0.75\}\)
   at fixed LR; (b) small sweep on **`r_max`** \(\in \{1.5, 2.0, 3.0\}\) for
   mutrust/muscale with monitoring of **fraction of steps at bound**; (c) keep
   **`orscale_muon_moonlight`** LR around **3e-3** where trust is visibly active.

---

## Related docs

- Main sweep narrative: [`report.md`](report.md)
- Analysis script: [`scripts/analyze_fineweb_trust_ratio.py`](../../scripts/analyze_fineweb_trust_ratio.py)
