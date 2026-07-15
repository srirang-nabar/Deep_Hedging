# Deep hedging under transaction costs: a stochastic control study

## The question

Discrete Black-Scholes delta hedging is the canonical answer to "how do I
neutralize the risk of a sold option," and with zero trading frictions it is
essentially unbeatable — its error vanishes like n^(−1/2) in the rebalance
count (we verify this in `notebooks/02_baselines.ipynb`). With proportional
transaction costs it over-trades catastrophically. Classical fixes exist —
Leland's volatility adjustment (1985) and the Whalley–Wilmott asymptotic
no-trade band (1997). The question: can a small neural network policy,
trained by direct differentiation through simulated P&L, beat the classical
fixes *when they are tuned to their best* — and does it rediscover the
structure theory says is optimal?

## Method

- **Market**: GBM (σ=20%), 3-month ATM call, 63 daily rebalances, S0=K=100;
  the hedger sells at the Black-Scholes premium and trades the underlying
  with proportional costs ∈ {0, 5, 20, 50} bps. Terminal P&L = premium +
  trading gains − costs − payoff (exact accounting identity, property-tested).
- **Objective**: J = mean(loss) + CVaR95(loss), identical for everyone —
  classical baselines are grid-search calibrated on it per cost level
  (baseline fairness protocol), and the network trains on it.
- **Policy**: MLP (4→32→32→1, ~1.3k params) mapping (log-moneyness, time
  fraction, current holding, EWMA realized-vol ratio) → position ∈ [0,1].
  Trained by backprop through the simulated P&L — the simulator is
  differentiable, so no policy-gradient machinery (see interview_qa.md).
  5 seeds per cost level; published numbers are across-seed means. Early
  stopping on VAL only; TEST is hash-fingerprinted and guarded in code.
- **Verification-first design**: every published number lives in CLAIMS.md
  with the notebook cell and pytest test that assert it; artifacts are
  SHA-256 frozen in a manifest; three reproduction tiers (verify ≤ minutes,
  recompute exactly, retrain statistically) in REPRODUCING.md. All three
  hypotheses were pre-registered in HYPOTHESES.md before their experiments
  ran, with quantified thresholds.

## Results

**Frozen baseline table** (100k TEST paths; CVaR95 of loss / mean turnover):

| Cost | Naive delta | Leland (calib.) | WW band (calib.) | Learned (5-seed mean) |
| --- | --- | --- | --- | --- |
| 5 bps | 1.2365 / 3.51 | 1.0635 / 3.29 | 1.2177 / 2.90 | **1.0542** / 3.06 |
| 20 bps | 1.9683 / 3.51 | 1.6067 / 3.21 | 1.8006 / 2.27 | **1.5781** / 2.74 |
| 50 bps | 3.5041 / 3.51 | 2.7287 / 3.01 | 2.7940 / 1.85 | **2.5408** / 2.23 |

The learned policy has the lowest CVaR95 at every cost level and also wins
the shared mean–CVaR objective everywhere (e.g. 2.134 vs WW's 2.265 at 20
bps). Two classical findings worth keeping: naive delta's turnover is
cost-invariant by construction, which is why its tail grows fastest; and
calibrated Leland — not the band — was the strongest classical baseline on
pure tail risk at low costs.

**Pre-registered hypotheses — all three failed as written, each
differently:**

- **H1 (beats WW at equal-or-lower turnover): not supported.** The CVaR half
  held decisively (paired diff −0.2230, 95% CI [−0.2334, −0.2120] at 20
  bps); the turnover half failed (2.74 vs 2.27) — the policy buys tail
  relief with extra trading. The turnover clause was never implied by the
  training objective; lesson recorded, bet not reworded.
- **H2 (recovers the no-trade band): not supported.** Width monotonicity
  passed 5/5 seeds (hold region widens with cost: 0.0208 → 0.0234 → 0.0333)
  but IoU with the calibrated WW band failed hard (0.035–0.137 vs 0.5): the
  network learned band-*like* inertia, located elsewhere in state space.
- **H3 (advantage shrinks off-distribution but doesn't invert): not
  supported — favorably.** No inversion anywhere (all CIs < 0). But on
  block-bootstrapped NIFTY returns the advantage *widened* (−0.3754 vs
  −0.2230 on GBM; Heston shrank to −0.1634 as predicted) — plausibly the EWMA vol
  feature adapting to real volatility clustering while WW's fixed-σ band
  cannot.

**Inventory ablation (20 bps):** policies retrained without the
current-holding input score CVaR95 1.5831 ± 0.0182 vs the full policy's
1.5781 ± 0.0161 — statistically indistinguishable — but trade ~14% more
(turnover 3.1104 vs 2.7367). The inventory feature buys cost efficiency,
not tail risk: knowing the current position lets the policy skip needless
trades without giving up protection.

## Honest limitations

- **Reproducibility is two different promises.** Committed weights
  re-evaluate to published numbers at 1e-6 on any CPU (exact); retraining
  reproduces only statistically, inside the ±4-across-seed-σ bands of
  `results/tolerances.json` — floating-point nondeterminism across hardware
  makes bit-exact retraining impossible, and we say so rather than pretend.
- **Training convergence was audited, not assumed — and is still bounded.**
  The first 50-epoch runs were quietly undertrained (caught because the
  policy lost to a one-parameter baseline on its own objective); after
  retraining at 120 epochs, probes measured residual headroom of
  +0.005–0.016 in J per 40 extra epochs — inside the across-seed spread, so
  we stopped, but the committed weights are near-converged, not optimal.
- **Where the learned policy did not win:** it never beat WW on turnover;
  H2's overlap test shows it does not implement the theoretically optimal
  band; and off-distribution its across-seed spread grows ~15x (±0.25 on
  NIFTY vs ±0.016 on GBM) — on real data, the seed you get matters far more.
- **Scope**: one option, one strike, one maturity, proportional costs only,
  no market impact, risk-neutral drift (μ=0). The NIFTY test resamples
  history (stationary block bootstrap, block-length sensitivity reported)
  and is not a live-market claim.
