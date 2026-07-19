# Reviewer summary — Option Hedging Optimization (Stochastic Control, Deep Learning)

**One paragraph.** When you sell an option, Black-Scholes theory says how to hedge it — but only in a
frictionless world; with transaction costs the textbook recipe over-trades. This project trains a small
(1.3k-parameter) neural policy by differentiating directly through simulated P&L on a mean–CVaR(95%)
objective, and compares it against the classical fixes (Leland 1985, Whalley–Wilmott 1997) *tuned to
their best on the same objective and data* — under the training model (GBM), under stochastic volatility
(Heston), and on block-bootstrapped real NIFTY returns. Hypotheses were pre-registered with quantified
thresholds; every public number has a CLAIMS.md row tied to a test and notebook cell.

**Findings:**

| Question | Result |
| -------- | ------ |
| Does the learned policy beat calibrated classical hedges? | **Lowest CVaR(95%) at every cost level** — 9–13% below calibrated Whalley–Wilmott, 1–7% below calibrated Leland (100k frozen test paths) — though at higher turnover than WW (disclosed; the pre-registered "beats at equal turnover" conjunction failed) |
| Does it survive model misspecification? | Yes — full advantage retained on Heston (−0.163) and **widened on real NIFTY data (−0.375)**, traced to an EWMA realized-vol feature static bands can't exploit |
| Does it rediscover the theoretical no-trade band? | Partly — hold-region widens with cost (5/5 seeds) but the pre-registered "recovers the WW band" claim was **rejected** (IoU ≤ 0.14) and published as a negative |

**How to review quickly (~5 min):**

1. Open `notebooks/05_analysis.ipynb` (band-recovery chart, misspecification results) and
   `notebooks/03_baseline_results.ipynb` (the headline table) — both commented, pre-executed,
   and end with asserts against CLAIMS.md.
2. Optional: `uv sync --frozen && uv run pytest -q` (65 tests, ~1 min); committed weights reproduce
   published metrics to 1e-6 on any CPU (`results/fresh_machine_run.log`).

**Scope honesty:** vanilla European calls, proportional costs, simulation-trained (GBM); baselines
calibrated per cost level on the same objective — the comparison is against their best, not textbook
defaults.
