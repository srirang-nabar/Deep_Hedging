# Deep Hedging

Learned option-hedging policies vs. Black-Scholes delta under proportional
transaction costs — a stochastic control study. With zero costs, discrete
delta hedging is hard to beat; with realistic costs it trades too much. This
project trains a small feedforward policy by differentiating directly through
simulated P&L against a mean–CVaR objective, and compares it to *calibrated*
classical baselines (Leland, Whalley–Wilmott) on frozen Monte Carlo test
sets, under both the training model (GBM) and misspecified dynamics (Heston,
bootstrapped NIFTY returns).

## Headline numbers

| Cost (bps) | Strategy | CVaR(95%) | Turnover | vs. calibrated WW |
| ---------- | -------- | --------- | -------- | ----------------- |
| _filled at Stage 6 from CLAIMS.md; every number here has a green assert_ | | | | |

## How to verify

See [REPRODUCING.md](REPRODUCING.md). Short version: `uv sync --frozen`, run
the notebooks (they assert every number above), and
`uv run python -m deep_hedging.manifest verify`. Claims-to-evidence mapping
lives in [CLAIMS.md](CLAIMS.md); pre-registered hypotheses in
[HYPOTHESES.md](HYPOTHESES.md).
