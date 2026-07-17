# Option Hedging Optimization (Stochastic Control, Deep Learning)

*a.k.a. Deep Hedging*

Learned option-hedging policies vs. classical delta hedging under
proportional transaction costs — a stochastic control study. With zero
costs, discrete Black-Scholes delta hedging is hard to beat; with realistic
costs it trades too much. This project trains a ~1.3k-parameter feedforward
policy by differentiating directly through simulated P&L against a
mean–CVaR(95%) objective and compares it, on frozen 100k-path test sets,
against classical baselines *calibrated per cost level on the same objective
and data* (Leland 1985, Whalley–Wilmott 1997) — under the training model
(GBM) and under misspecification (Heston, block-bootstrapped NIFTY 50).

## Headline numbers

CVaR(95%) of terminal hedging loss (lower is better) / mean turnover, frozen
TEST set, calibrated baselines, learned = mean over 5 training seeds. Every
number is a row in [CLAIMS.md](CLAIMS.md) backed by an executable assert.

| Cost | Naive delta | Leland (calib.) | WW band (calib.) | Learned policy |
| --- | --- | --- | --- | --- |
| 5 bps | 1.2365 / 3.51 | 1.0635 / 3.29 | 1.2177 / 2.90 | **1.0542** / 3.06 |
| 20 bps | 1.9683 / 3.51 | 1.6067 / 3.21 | 1.8006 / 2.27 | **1.5781** / 2.74 |
| 50 bps | 3.5041 / 3.51 | 2.7287 / 3.01 | 2.7940 / 1.85 | **2.5408** / 2.23 |

Off-distribution (trained on GBM, 20 bps): the policy's paired CVaR
advantage over calibrated WW is −0.1634 on Heston and −0.3754 on NIFTY
block-bootstrap paths — it never inverts, and on real data it *grows*.

**All three pre-registered hypotheses came out "not supported" — each
informatively.** H1: the policy beats WW on tail risk (CI [−0.2334, −0.2120])
but at higher turnover, failing the pre-registered conjunction.
H2: its hold region widens with cost (5/5 seeds) but does not coincide with
the WW band (IoU ≤ 0.137 vs 0.5 threshold) — band-*like*, not *the* band.
H3: the advantage was predicted to shrink off-distribution; on NIFTY it
widened instead. Details in [HYPOTHESES.md](HYPOTHESES.md) and
[report/report.md](report/report.md).

## How to verify

```bash
uv sync --frozen
uv run python -m deep_hedging.reproduce --tier 1   # minutes: hash-verify artifacts, run all asserting notebooks
uv run python -m deep_hedging.reproduce --tier 2   # recompute results from committed inputs, exact tolerances
uv run python -m deep_hedging.reproduce --tier 3   # hours: retrain from scratch into published tolerance bands
```

See [REPRODUCING.md](REPRODUCING.md) for the tier contract (exact vs.
statistical reproduction), [CLAIMS.md](CLAIMS.md) for the number→evidence
map, and [HYPOTHESES.md](HYPOTHESES.md) for the pre-registered bets and
verdicts. Test/train hygiene: TRAIN/VAL/TEST path pools are SHA-256
fingerprinted and disjoint; TEST access raises without a sign-off flag; a
leak test proves committed weights never saw TEST.
