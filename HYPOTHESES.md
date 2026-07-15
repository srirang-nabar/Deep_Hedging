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

- Metric and thresholds (quantified 2026-07-15, BEFORE the band analysis ran;
  the committed weights had not been probed on this grid when these were set):
  - Grid: spot S ∈ [85, 115] in 61 steps, inventory h ∈ [0, 1] in 41 steps,
    at mid-life τ = 0.125 (T/2), vol feature at its GBM resting value
    (vol_ratio = 0). Per policy seed and cost level, the empirical no-trade
    region is {(S, h): |policy(S, h) − h| < ε} with ε = 0.01 shares.
  - The reference region is the calibrated WW band at the same cost level:
    {(S, h): |h − δ(S)| ≤ H(S)} with H from the calibrated risk aversion.
  - (a) Overlap: IoU = |A∩B| / |A∪B| over grid cells, averaged across the 5
    seeds. Threshold: mean IoU ≥ 0.5 at every cost level in {5, 20, 50} bps.
  - (b) Width: per S, the h-measure of the no-trade region; band width =
    mean over the S grid. Threshold: across-seed mean width strictly
    increasing in cost over {5, 20, 50} bps AND at least 4 of 5 seeds
    individually strictly increasing.
  - "Recovers the band" enters CLAIMS.md only if BOTH (a) and (b) pass.
- Verdict (Stage 5 gate, 2026-07-15): **not supported.** Monotonicity passed
  cleanly — across-seed mean hold-region widths 0.0208 → 0.0234 → 0.0333
  across {5, 20, 50} bps, strictly increasing for 5 of 5 seeds. But IoU
  failed decisively: 0.035 / 0.058 / 0.137 vs the 0.5 threshold. The policy
  learned band-LIKE inertia that strengthens with cost, but its near-fixed-
  point region is far narrower than the calibrated WW band and does not
  coincide with it in the (spot, inventory) plane. "Recovers the band" does
  not enter CLAIMS.md. See results/band_analysis.json.

## H3 — Advantage degrades but does not invert under misspecification

When the policy is trained on GBM paths and evaluated on (a) Heston TEST
paths and (b) block-bootstrapped NIFTY returns, its advantage over the
calibrated WW baseline shrinks but does not invert (the learned policy does
not become worse than the baseline).

- Test (rules quantified 2026-07-15, before any misspecified evaluation ran):
  at 20 bps, paired across-seed CVaR(95%) difference (policy − calibrated WW)
  with a seeded 500-resample bootstrap CI, on two primary misspecified sets:
  (a) Heston, literature defaults, dedicated seed 404, 100k paths;
  (b) NIFTY 50 stationary block bootstrap, block length 10 (5 and 20 as
  sensitivity), seed 505, 100k paths, S0 rebased to 100. The hedger's model
  stays GBM(σ=0.2) throughout — that is the misspecification.
  - "Does not invert": CI upper bound < 0 on both primary sets.
  - "Shrinks": diff mean on each primary set is less negative than the GBM
    TEST diff (−0.2230).
  - Supported iff both conditions hold on both primary sets.
- Verdict (Stage 5 gate, 2026-07-15): **not supported — by a favorable
  violation.** "Does not invert" passed on every set (policy beats WW with
  CI < 0 everywhere: Heston −0.1634 [−0.1782, −0.1492]; NIFTY block-10
  −0.3754 [−0.4134, −0.3408]; blocks 5/20 similar). "Shrinks" failed: the
  advantage shrank on Heston as predicted but WIDENED on NIFTY (−0.375 vs
  GBM's −0.223) — plausibly the EWMA vol feature adapting to real vol
  clustering while WW's fixed-sigma band cannot. Caveat recorded: across-seed
  CVaR spread grows an order of magnitude under distribution shift (±0.25 on
  NIFTY vs ±0.016 on GBM). See results/misspecification.json.
