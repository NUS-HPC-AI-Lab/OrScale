# FineWeb-Edu small_125m: post-fix sweep leaderboard

Sorted by `final_val_loss` (mean of last 3 logged val checkpoints).

| tag | optimizer | lr | final_train | final_val | best_val | bump | grad_post_warmup_mean | grad_post_warmup_sat_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | muon | 0.005 | 3.2575 | 3.2509 | 3.2506 | no | - | - |
| baseline | muon | 0.01 | 3.2320 | 3.2242 | 3.2239 | no | - | - |
| baseline | muon | 0.02 | 3.2183 | 3.2111 | 3.2106 | no | - | - |
| baseline | muon | 0.04 | 3.2285 | 3.2214 | 3.2207 | no | - | - |
| new | muon_moonlight | 3e-4 | 3.2908 | 3.2841 | 3.2839 | no | 0.465 | 0.030 |
| new | muon_moonlight | 1e-3 | 3.2399 | 3.2319 | 3.2317 | no | 0.258 | 0.000 |
| new | muon_moonlight | 3e-3 | 3.2439 | 3.2372 | 3.2370 | no | 0.335 | 0.005 |
| new | muon_moonlight | 1e-2 | 4.6035 | 4.6262 | 4.4595 | YES | 9.216 | 0.996 |
| old | muon_moonlight | 0.005 | 4.3126 | 4.3237 | 4.0135 | YES | - | - |
| old | muon_moonlight | 0.01 | 4.6406 | 4.6690 | 4.4868 | YES | - | - |
| old | muon_moonlight | 0.02 | 4.7244 | 4.7518 | 4.7513 | YES | - | - |
| old | muon_moonlight | 0.04 | 4.8954 | 4.9272 | 4.9263 | YES | - | - |
| new | muscale | 1e-4 | 3.3507 | 3.3456 | 3.3454 | no | 0.763 | 0.131 |
| new | muscale | 3e-4 | 3.2669 | 3.2604 | 3.2602 | no | 0.375 | 0.009 |
| new | muscale | 5e-4 | 3.2478 | 3.2413 | 3.2411 | no | 0.288 | 0.000 |
| new | muscale | 1e-3 | 3.2378 | 3.2305 | 3.2303 | no | 0.252 | 0.000 |
| new | mutrust | 0.005 | 3.2418 | 3.2339 | 3.2337 | no | 0.264 | 0.000 |
| new | mutrust | 0.01 | 3.2290 | 3.2215 | 3.2212 | no | 0.289 | 0.000 |
| new | mutrust | 0.02 | 3.2238 | 3.2164 | 3.2160 | no | 0.360 | 0.038 |
| new | mutrust | 0.04 | 3.7306 | 3.7332 | 3.7326 | YES | 0.414 | 0.109 |
| old | mutrust | 0.005 | 4.4239 | 4.4408 | 3.9093 | YES | - | - |
| old | mutrust | 0.01 | 4.8618 | 4.8943 | 4.6211 | YES | - | - |
| old | mutrust | 0.02 | 4.7959 | 4.8256 | 4.7724 | YES | - | - |
| old | mutrust | 0.04 | 5.2762 | 5.3139 | 5.3129 | YES | - | - |
| new | orscale_muon_moonlight | 3e-4 | 3.3501 | 3.3446 | 3.3445 | no | 0.761 | 0.129 |
| new | orscale_muon_moonlight | 1e-3 | 3.2619 | 3.2556 | 3.2554 | no | 0.357 | 0.006 |
| new | orscale_muon_moonlight | 3e-3 | 3.2310 | 3.2232 | 3.2229 | no | 0.248 | 0.000 |
| new | orscale_muon_moonlight | 1e-2 | 4.6521 | 4.6790 | 4.1625 | YES | 6.475 | 0.999 |
| new | orscale_muon_moonlight_calibrated | 1e-3 | 3.2337 | 3.2264 | 3.2261 | no | 0.197 | 0.004 |
| old | orscale_muon_moonlight | 0.005 | 3.2186 | 3.2117 | 3.2113 | no | - | - |
| old | orscale_muon_moonlight | 0.01 | 4.7287 | 4.7549 | 3.8971 | YES | - | - |
| old | orscale_muon_moonlight | 0.02 | 5.8882 | 5.9297 | 4.3660 | YES | - | - |
| old | orscale_muon_moonlight | 0.04 | 6.1719 | 6.2164 | 4.6805 | YES | - | - |
