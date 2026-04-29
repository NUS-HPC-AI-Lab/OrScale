## MuScale follow-up sweep (2026-04-29)

`muscale` is a fresh OrScale variant (Nesterov momentum, `RMS(M_hat)`
denominator, Moonlight `0.2*sqrt(max(m,n))` shape rescaling, no partial
exponent -- i.e. `muscale_alpha` with `alpha=1.0`). The first CIFAR sweep on
`{0.005, 0.01, 0.02, 0.04}` found a flat surface with the best cell at the
upper boundary. This report now uses the follow-up upward bracket:

- **Sweep dir**: `sweeps/20260429_115704/` (12 runs, all completed)
- **LR grid**: `{0.04, 0.06, 0.08, 0.12}`
- **Seeds**: `42, 43, 44`
- **Settings**: config defaults from `configs/cifar10_davidnet.yaml`
  (`r_min=0.5`, `r_max=1.5`, `momentum=0.95`, `weight_decay=5e-4`,
  `eps=1e-6`, `ns_iters=5`)

### Headline result

| LR | val_top1 (mean ± std, last 3 epochs) | final val_top1 (mean ± std) | wallclock (s) |
| --- | --- | --- | --- |
| 0.04 | 93.75 ± 0.08 | 93.77 ± 0.10 | 94.6 ± 3.3 |
| 0.06 | **93.83 ± 0.17** | **93.88 ± 0.18** | 110.5 ± 16.7 |
| 0.08 | 93.66 ± 0.22 | 93.67 ± 0.26 | 97.1 ± 12.8 |
| 0.12 | 93.56 ± 0.20 | 93.57 ± 0.22 | 119.0 ± 6.3 |

The upward bracket confirms that the optimum moved off the original upper
edge and peaks at `lr=0.06`, not `0.04`. Going higher degrades smoothly:
`0.08` is already back at the old-grid level, and `0.12` is clearly worse.
There is no sign of catastrophic instability on CIFAR, just ordinary
over-stepping once the LR gets too large.

### Where it lands in the head-to-head

In the full 9-optimizer ranking, `muscale@0.06` is rank 5 with
93.83% ± 0.17 last-3-epoch val_top1. It is essentially tied with
`mutrust@0.005` (93.84% ± 0.15) and now sits above both
`muon_moonlight@0.01` (93.75%) and vanilla `muon@0.04` (93.70%).

It still trails the OrScale trio at the top:

- `orscale_muon@0.02`: 94.06% ± 0.09
- `orscale_muon_wd@0.02`: 94.05% ± 0.08
- `orscale_muon_moonlight@0.02`: 94.05% ± 0.12

So the new grid improves `muscale` by ~0.08 points over the first
`0.04` result, but it does **not** close the ~0.2 point gap to the best
OrScale variants on CIFAR-10 / DavidNet.

### Recommendations

1. **Use `lr=0.06` as the CIFAR muscale default** for DavidNet-style runs.
   The bracket `{0.04, 0.06, 0.08, 0.12}` now contains the optimum cleanly.
2. **Do not widen further upward** unless another architecture suggests it.
   `0.08` and `0.12` both lose accuracy relative to `0.06`.
3. **Do not change `r_min=0.5`, `r_max=1.5` based on CIFAR.** The grid is
   stable across all four LRs, and the limitation is final accuracy rather
   than visible training instability.

### Artifacts

| file | what |
| --- | --- |
| `sweeps/20260429_115704/sweep_config.json` | exact follow-up grid |
| `sweeps/20260429_115704/name=muscale_lr=*_seed=*.log` | per-run stdout |
| `runs.csv`, `epochs.csv`, `steps.csv` | tidy data (new muscale rows included) |
| `summary_by_opt_lr.md` / `.csv` | leaderboard with the upward muscale grid |
| `plots/curves_val_top1.png` | best-LR head-to-head (muscale = olive) |
| `plots/lr_sensitivity_val_top1.png` | LR sweep including the new muscale bracket |
| `plots/variance_best_lr_val_top1.png` | per-seed bars at each best LR |
| `plots/wallclock_seconds.png` | wall-clock comparison |
