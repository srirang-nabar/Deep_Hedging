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

Honest note (Stage 3): on CVaR(95%) alone, calibrated Leland slightly beats
the WW band at every positive cost level; WW wins on the combined mean–CVaR
objective and on turnover. "The band dominates on tail risk" would be an
overclaim and is not made anywhere.
