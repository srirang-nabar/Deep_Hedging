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
- Verdict (Stage 4 gate): _pending_

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
