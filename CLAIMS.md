# Claims register

One row per number that appears in the README or a resume bullet. A claim may
not appear anywhere public unless its row here names the notebook cell and the
pytest test that assert it. Values are frozen when the owning stage's gate
passes. Metrics are computed on the frozen TEST path set (see
`results/path_sets.json`); strategies are calibrated per the baseline
fairness protocol (`results/baseline_calibration.json`).

| ID | Claim | Value | Verified by (notebook) | Verified by (test) | Stage |
| -- | ----- | ----- | ---------------------- | ------------------ | ----- |
| BT20-DELTA-CVAR95 | CVaR(95%) of naive BS delta at 20 bps | 1.9683 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT20-LELAND-CVAR95 | CVaR(95%) of calibrated Leland at 20 bps | 1.6067 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT20-WW-CVAR95 | CVaR(95%) of calibrated Whalley–Wilmott at 20 bps | 1.8006 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-DELTA-CVAR95 | CVaR(95%) of naive BS delta at 50 bps | 3.5041 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-LELAND-CVAR95 | CVaR(95%) of calibrated Leland at 50 bps | 2.7287 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-WW-CVAR95 | CVaR(95%) of calibrated Whalley–Wilmott at 50 bps | 2.7940 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-DELTA-TURNOVER | Mean turnover of naive BS delta at 50 bps | 3.5100 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-LELAND-TURNOVER | Mean turnover of calibrated Leland at 50 bps | 3.0146 | 03_baseline_results | test_claims_match_frozen_table | 3 |
| BT50-WW-TURNOVER | Mean turnover of calibrated Whalley–Wilmott at 50 bps | 1.8481 | 03_baseline_results | test_claims_match_frozen_table | 3 |

| LP5-CVAR95-MEAN | CVaR(95%) of learned policy at 5 bps, mean over 5 seeds | 1.1418 | 04_learned_policy | test_published_metrics_reproduce | 4 |
| LP20-CVAR95-MEAN | CVaR(95%) of learned policy at 20 bps, mean over 5 seeds | 1.6477 | 04_learned_policy | test_published_metrics_reproduce | 4 |
| LP50-CVAR95-MEAN | CVaR(95%) of learned policy at 50 bps, mean over 5 seeds | 2.5870 | 04_learned_policy | test_published_metrics_reproduce | 4 |
| LP20-TURNOVER-MEAN | Mean turnover of learned policy at 20 bps, mean over 5 seeds | 2.7452 | 04_learned_policy | test_published_metrics_reproduce | 4 |
| H1-CVAR-DIFF-MEAN | Paired CVaR(95%) difference, policy − calibrated WW, 20 bps | -0.1534 | 04_learned_policy | test_h1_verdict_recorded_and_consistent | 4 |

Honest note (Stage 4): H1 as pre-registered is **not supported** — the
policy reduces CVaR(95%) vs calibrated WW at every cost level (CI excludes
zero) but at *higher* turnover, failing the pre-registered conjunction. On
pure CVaR(95%), calibrated Leland still beats the policy at 5 and 20 bps
(not at 50). Any public claim must carry these qualifiers.

Honest note (Stage 3): on CVaR(95%) alone, calibrated Leland slightly beats
the WW band at every positive cost level; WW wins on the combined mean–CVaR
objective and on turnover. "The band dominates on tail risk" would be an
overclaim and is not made anywhere.
