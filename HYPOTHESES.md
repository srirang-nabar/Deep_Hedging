# Pre-registered hypotheses

Registered 2026-07-14, before any experiment has run. Verdicts are filled in
at the gates named below and are never edited afterwards; a negative result is
recorded, not reworked.

## H1 — Learned policy beats calibrated Whalley–Wilmott at 20 bps

At 20 bps proportional transaction costs, a learned hedging policy reduces
CVaR(95%) of the terminal hedging-error distribution relative to the
**calibrated** Whalley–Wilmott baseline (calibrated per the baseline fairness
protocol in plan.md Stage 2), at equal or lower turnover.

- Test: paired comparison on common TEST paths, across-seed mean over ≥5
  training seeds, bootstrap CI on the CVaR difference. Supported only if the
  CI excludes zero *and* mean turnover is not higher.
- Verdict (Stage 4 gate, 2026-07-14): **not supported** — an honest negative
  on the conjunction. The CVaR half held decisively: across-seed mean
  CVaR(95%) 1.5781 ± 0.0161 vs calibrated WW 1.8006, paired-bootstrap
  difference −0.2230 with 95% CI [−0.2334, −0.2120], entirely below zero.
  The turnover half failed: policy turnover 2.737 vs WW 2.271 — the learned
  policy buys its tail-risk reduction with *more* trading, not less. It
  beats every calibrated baseline on the shared mean–CVaR objective at every
  cost level (e.g. 2.134 vs WW 2.265 at 20 bps). Verdict computed on the
  120-epoch retrained weights (training audit, same date); the pre-registered
  turnover condition was arguably never implied by the shared objective, but
  it stands as written. See results/learned_policy_results.json.

## H2 — No-trade-band structure, width increasing in cost

The learned policy exhibits no-trade-band structure whose width increases
with the cost level.

- Metric (committed in words now; quantified thresholds set in Stage 5 before
  that analysis runs): over a grid of (spot, time, inventory) states, the
  empirical no-trade region is the set of states where |Δposition| < ε.
  Report (a) IoU overlap with the calibrated WW band and (b) fitted band
  width per cost level in {5, 20, 50} bps with a monotonicity trend test.
  "Recovers the band" enters CLAIMS.md only if both pass their pre-set
  thresholds.
- Verdict (Stage 5 gate): _pending_

## H3 — Advantage degrades but does not invert under misspecification

When the policy is trained on GBM paths and evaluated on (a) Heston TEST
paths and (b) block-bootstrapped NIFTY returns, its advantage over the
calibrated WW baseline shrinks but does not invert (the learned policy does
not become worse than the baseline).

- Test: same paired-CI machinery as H1 on each misspecified test set;
  degradation reported with across-seed CIs.
- Verdict (Stage 5 gate): _pending_
