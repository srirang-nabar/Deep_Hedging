# Deep Hedging — Staged Implementation Plan

Learned option-hedging policies vs. Black-Scholes delta under transaction costs. Full research rationale: [../placement_projects/01_deep_hedging.md](../placement_projects/01_deep_hedging.md).

**How to use this file:** each stage is a self-contained work order with a definition of done. Stages are ordered; do not start a stage until the previous stage's gate passes. Gates are executable: `uv run pytest -m gate_stageN` must be green, and the stage's notebook must run top-to-bottom. Tick checkboxes as work completes and record headline numbers in the Results Log at the bottom.

**Ground rules (apply to every stage):**

- Package manager is `uv` only: `uv add`, `uv run`, `uv run pytest`. Never pip.
- No git commits or pushes — the user handles all version control. (Files that *should* be committed, like `uv.lock` and artifact manifests, are simply left in place for the user.)
- Every random process takes an explicit seed; every experiment writes config + seed + outputs to `results/`.
- Framing rule for all prose/docstrings/reports: this is a *stochastic control / derivatives* project. Never describe it as "RL for trading."

## Reproducibility & Verification Charter

External volunteers will verify the CV claims on their own machines. Design for them from day 1.

**Three tiers of reproduction (documented in `REPRODUCING.md`):**

- **Tier 1 — Verify claims (≤10 min, CPU laptop):** `uv sync --frozen`, then run the numbered notebooks. Notebooks load committed artifacts (trained weights, frozen tables, data snapshots) and **assert** every headline number, not just print it.
- **Tier 2 — Recompute results (≤1 hr, CPU):** re-run evaluation from committed weights and path-set seeds; regenerate every table and figure from scratch. `uv run python -m deep_hedging.reproduce --tier 2`.
- **Tier 3 — Full retrain (hours, CPU-feasible; GPU optional):** retrain policies from scratch with documented seeds. Because floating-point nondeterminism across hardware makes bit-exact retraining impossible, Tier 3 promises **statistical** reproduction: retrained metrics must land inside the across-seed tolerance bands published in `results/tolerances.json`. This distinction (exact artifact reproduction vs. statistical process reproduction) is stated explicitly in `REPRODUCING.md`.

**Artifact policy:**

- Trained weights saved as `state_dict` + a JSON sidecar (architecture, training config, seed, torch version, git-diff-free hash of training code) under `results/weights/`. Weights must load and evaluate **on CPU** with pinned torch; a dedicated test proves saved weights reproduce published metrics to numerical tolerance.
- All external data is snapshotted into the repo (`data/` — NIFTY daily closes are a few hundred KB) with SHA256 recorded in `results/MANIFEST.sha256`. No volunteer ever depends on yfinance being up or unchanged.
- Environment: `uv.lock` + `.python-version` are the contract; `REPRODUCING.md` opens with `uv sync --frozen`. Torch pinned exactly.
- `CLAIMS.md`: one row per resume/README claim → the notebook cell and pytest test that verifies it. A claim without a verifying artifact may not appear in the README.
- Smoke configs: every experiment has a `--smoke` variant (small paths/epochs) that runs in minutes and exercises the identical code path; full configs reproduce the published numbers.

**Notebook plan (`notebooks/`, executed outputs saved in place, numbered by stage):**

| Notebook | Contents (for a skimming volunteer) |
| --- | --- |
| `01_simulators.ipynb` | What GBM/Heston are, sample paths, moment checks vs. closed form with assertions |
| `02_baselines.ipynb` | BS delta, Leland, Whalley–Wilmott explained; the zero-cost convergence plot; golden-value assertions |
| `03_baseline_results.ipynb` | The frozen baseline table (strategies × cost levels) regenerated + asserted; why WW beats naive delta as costs rise |
| `04_learned_policy.ipynb` | Loads committed weights on CPU, re-evaluates on the frozen test paths, asserts headline CVaR/turnover numbers, shows training curves |
| `05_analysis.ipynb` | Band-recovery money chart, band-width-vs-cost monotonicity test, inventory ablation, misspecification results — each with an assert against `CLAIMS.md` values |

Each notebook: a 3-sentence "what you are looking at" intro per section, and a final cell `verify_claims("notebook_id")` that cross-checks every number shown against `CLAIMS.md`.

**Test policy — two kinds, both gate progression:**

- *Coding tests:* unit + property tests for every module; determinism tests (same seed → identical output on CPU); serialization round-trips.
- *Statistical tests:* acceptance tests with **derived** tolerances (Monte Carlo CIs, not magic numbers) — e.g., simulated GBM mean within 4 SE of closed form at n=100k. Statistical gates are pytest tests with fixed seeds so they are deterministic in CI while still testing distributional claims.
- Marker convention: `@pytest.mark.gate_stage1` etc.; `uv run pytest -m gate_stageN` is the literal gate command.

## Stage 0 — Environment & protocol

- [x] `uv add numpy scipy torch matplotlib pandas` and `uv add --dev pytest hypothesis jupyter nbconvert`
- [x] Pin torch exactly in `pyproject.toml`; verify `uv.lock` exists (left for user to commit)
- [x] Package skeleton under `src/deep_hedging/`: `simulate.py`, `pricing.py`, `baselines.py`, `evaluate.py`, `policy.py`, `train.py`, `reproduce.py` (empty modules with responsibility docstrings)
- [x] Repro skeleton: `REPRODUCING.md` (tier structure, filled as stages complete), `CLAIMS.md` (empty table), `results/MANIFEST.sha256` tooling (tiny helper that hashes artifacts), pytest markers registered in `pyproject.toml`
- [x] `HYPOTHESES.md` (dated) — pre-register before any experiment runs:
  - H1: at 20 bps proportional costs, a learned policy reduces CVaR(95%) of terminal hedging error vs. the *calibrated* Whalley–Wilmott baseline at equal or lower turnover
  - H2: the learned policy exhibits no-trade-band structure whose width increases with the cost level (quantified metric defined in Stage 5, committed here in words)
  - H3: advantage shrinks under model misspecification (train GBM → test Heston / bootstrapped NIFTY) but does not invert
- [x] `README.md` stub: one-paragraph description + empty headline-numbers table (TRIAD-RL house style) + "how to verify" pointer to REPRODUCING.md

**Gate (`gate_stage0`):** `uv run pytest` runs; `uv sync --frozen` succeeds in a scratch copy of the folder (proves the lockfile is self-sufficient).

## Stage 1 — Market simulators

- [x] GBM path simulator: vectorized (n_paths × n_steps), explicit dtype/seed/generator object
- [x] Heston simulator (full truncation Euler; literature defaults κ=2, θ=0.04, σ_v=0.3, ρ=−0.7)
- [x] Coding tests:
  - [x] Seed determinism: same seed → bit-identical paths on CPU (this test is the volunteer's guarantee)
  - [x] Shape/dtype contracts; generator isolation (two simulators with different seeds don't interact)
- [x] Statistical tests (tolerances derived from MC standard errors, documented in the test docstrings):
  - [x] GBM terminal mean & variance within 4 SE of closed form at n=100k; log-returns pass normality at a pre-set n
  - [x] Discounted-price martingale property under the risk-neutral measure
  - [x] Heston: variance non-negative; long-run mean variance → θ within CI; leverage effect sign (corr(dW_S, dW_v) < 0)
- [x] `notebooks/01_simulators.ipynb` written and executed

**Gate (`gate_stage1`):** all Stage 1 tests green; notebook runs end-to-end via `uv run jupyter execute`.

## Stage 2 — Pricing & classical baselines

- [x] Black-Scholes price + delta (vectorized)
- [x] Hedging engine: given any strategy (callable state → position), simulate discrete rebalancing with proportional costs; return terminal hedging-error distribution
- [x] Baselines: BS delta; Leland-adjusted delta; Whalley–Wilmott no-trade band
- [x] **Baseline fairness protocol (critic fix — see Critic's Review):** WW band width and Leland parameters are *calibrated* per cost level by grid search on the same mean–CVaR objective and the same training paths the learned policy will use. The published comparison is against calibrated baselines, never textbook defaults. Calibration grids + chosen params saved to `results/baseline_calibration.json`.
- [x] Coding tests:
  - [x] BS golden values (published table values); put-call parity as a property test over random (S, K, T, σ, r)
  - [x] Hedging engine accounting identity: cash + stock + option legs decompose and sum exactly (property test on random strategies)
  - [x] Zero-cost, zero-rebalance-limit equivalences (engine reduces to analytic cases)
- [x] Statistical tests:
  - [x] **The interview test:** with zero costs, BS-delta hedging error → 0 as rebalance frequency grows; fitted log-log slope = −0.5 within CI
  - [x] Ordering sanity at 50 bps: calibrated WW ≤ naive delta on CVaR(95%) (strict inequality expected; test at 95% bootstrap confidence)
- [x] `notebooks/02_baselines.ipynb` written and executed (includes `results/convergence_check.png` — a keeper for the report)

**Gate (`gate_stage2`):** all tests green; convergence plot exists; calibration JSON exists.

## Stage 3 — Evaluation harness & frozen test sets

- [x] Path-set discipline (critic fix): three disjoint seeded path sets — TRAIN (policy training), VAL (model selection, early stopping, any tuning), TEST (final numbers; **touched only in Stage 5 sign-off runs and notebooks**). Seeds and hashes of all three recorded in `results/MANIFEST.sha256`.
- [x] Monte Carlo evaluation: 100k TEST paths per (strategy × cost level), cost levels {0, 5, 20, 50} bps
- [x] Metrics: mean, std, CVaR(95%), CVaR(99%), turnover; bootstrap CIs on CVaR (seeded bootstrap)
- [x] Results table generator → `results/baseline_table.md` + machine-readable `results/baseline_table.json`
- [x] Coding tests: metric unit tests on hand-computable toy distributions; bootstrap determinism under fixed seed; table generator round-trip (json ↔ md agree)
- [x] Statistical test: all strategies statistically indistinguishable at zero cost (they should coincide up to MC noise — a strong harness check)
- [x] `notebooks/03_baseline_results.ipynb` written and executed; baseline rows entered into `CLAIMS.md`

**Gate (`gate_stage3`):** baseline table frozen (hash in manifest); later stages compare against this exact artifact.

## Stage 4 — Learned policy

- [x] Feedforward policy: state = (log-moneyness, time-to-expiry, current holding, running vol estimate) → position; small MLP first
- [x] Training by direct differentiation through simulated P&L on TRAIN paths (the simulator is differentiable — no policy-gradient machinery); document *why* in the report
- [x] Objective: mean–CVaR(95%) tradeoff, λ-parameterized; train per cost level
- [x] **Multi-seed protocol (critic fix):** ≥5 training seeds per cost level. Published number = mean ± std across seeds evaluated on TEST; no per-seed cherry-picking. Across-seed bands become `results/tolerances.json` (the Tier 3 contract).
- [x] Training hygiene: early stopping on VAL CVaR only; loss curves saved; TEST never touched during development (enforced in code: evaluation module refuses TEST path-set ID unless `final=True` flag set by the sign-off script)
- [x] Artifacts: per-seed `state_dict` + JSON sidecar under `results/weights/`; hashes in manifest
- [x] Coding tests:
  - [x] Weight round-trip: save → load on CPU → identical outputs on a probe batch (tolerance 0)
  - [x] **Published-metrics reproduction test:** loading committed weights and evaluating on TEST reproduces the numbers in `CLAIMS.md` to 1e-6 (this single test is the core volunteer guarantee)
  - [x] Gradient sanity: loss decreases on a tiny overfit problem; gradients flow to all parameters
  - [x] Leak test: assert TRAIN/VAL/TEST path-set hashes are pairwise distinct and the training loop only ever received TRAIN/VAL IDs
- [x] Statistical test (the H1 gate): at 20 bps, across-seed mean CVaR(95%) vs. calibrated WW — paired comparison on common paths, bootstrap CI
- [x] `notebooks/04_learned_policy.ipynb` written and executed (loads committed weights, asserts headline numbers)

**Gate (`gate_stage4`):** H1 has a verdict — supported *or* an honest documented negative. Either proceeds; silent regressions don't. All weight artifacts load on CPU in the notebook.

## Stage 5 — Analysis & stress tests

- [ ] **Band-recovery, quantified (critic fix):** define the metric up front — for a grid of (S, t, inventory), compute the empirical no-trade region of the learned policy (states where |Δposition| < ε); report (a) overlap (IoU) with the calibrated WW band, (b) fitted band width per cost level with a monotonicity test (Page's trend test or isotonic-fit R²) across {5, 20, 50} bps. "Recovers the band" appears in claims only if both pass pre-set thresholds written in HYPOTHESES.md.
- [ ] Ablation: stateless policy (no inventory input) vs. full policy, same multi-seed protocol
- [ ] Misspecification: train GBM → evaluate on (a) Heston TEST paths, (b) block-bootstrapped NIFTY returns. NIFTY data snapshotted to `data/nifty_daily.csv` with source, date-range, and SHA256; block-length sensitivity (2–3 values) reported
- [ ] Optional (only if ahead of schedule): PPO comparison to demonstrate why direct differentiation was the right call
- [ ] Statistical tests: monotonicity gate; degradation quantified with across-seed CIs; H2/H3 verdicts recorded in HYPOTHESES.md
- [ ] `notebooks/05_analysis.ipynb` written and executed; all Stage 5 claims entered into `CLAIMS.md`

**Gate (`gate_stage5`):** H1–H3 all have numeric verdicts; band metric computed, not eyeballed.

## Stage 6 — Write-up & verification pack

- [ ] `README.md` headline table filled from `CLAIMS.md` (numbers must match assert values exactly)
- [ ] `report/report.md`: question → method → results → honest limitations (including: what exact vs. statistical reproducibility means here; where the learned policy did *not* win)
- [ ] `report/interview_qa.md`: anticipated grills (why CVaR not variance; Leland's argument; why differentiate through the sim rather than model-free RL; how baselines were calibrated fairly; what breaks under misspecification and why; how a volunteer verifies the claims)
- [ ] `REPRODUCING.md` finalized with measured wall-clock times per tier on a reference CPU
- [ ] **Fresh-machine dry run (final gate):** copy the folder to a clean directory (or container), `uv sync --frozen`, run Tier 1 end-to-end. Must pass with zero manual fixes. Record the transcript in `results/fresh_machine_run.log`.
- [ ] Resume bullets drafted with real numbers, framed per the framing rule; each bullet has a `CLAIMS.md` row

**Gate (`gate_stage6`):** fresh-machine Tier 1 passes; every README number has a green assert.

## Critic's Review — issues found and fixes baked in above

Adversarial pass performed 2026-07 from the perspective of a quant interviewer / skeptical volunteer:

| # | Issue a critic would raise | Fix (where) |
| --- | --- | --- |
| 1 | "You beat Black-Scholes delta" — against *textbook* baselines, i.e., a straw man. WW has a free risk-aversion parameter; untuned, it's easy to beat. | Baseline fairness protocol: calibrate WW/Leland per cost level on the same objective and data (Stage 2); headline comparisons are vs. *calibrated* baselines |
| 2 | Single training run → the reported number could be a lucky seed. | ≥5 seeds per cell; published number = across-seed mean ± std; tolerances.json makes this the Tier 3 contract (Stage 4) |
| 3 | Model selection on the evaluation set — subtle overfitting via repeated peeking. | TRAIN/VAL/TEST path-set discipline with hash-enforced separation and a code-level guard on TEST access (Stages 3–4) |
| 4 | "Recovers the no-trade band" is an eyeballed claim. | Quantified band metric: no-trade-region IoU + width-monotonicity trend test, thresholds pre-registered (Stage 5) |
| 5 | Weights trained on one machine won't reproduce bit-exact elsewhere; volunteers will call the retrain "failed." | Two-level guarantee: exact reproduction from committed CPU-loadable weights (tested to 1e-6), statistical reproduction for retraining within published tolerance bands; distinction documented in REPRODUCING.md |
| 6 | yfinance data drifts/disappears → misspecification results unreproducible. | NIFTY snapshot committed with SHA256 + source/date provenance (Stage 5) |
| 7 | CVaR(95%) on 100k paths has estimation error — is the win larger than the noise? | Bootstrap CIs on every CVaR; paired path-level comparisons; H1 gate requires CI-separated difference (Stages 3–4) |
| 8 | Volunteers on laptops can't wait hours to check anything. | Tier 1 ≤10 min from committed artifacts; smoke configs exercise full code paths in minutes; measured runtimes published (Charter, Stage 6) |
| 9 | Notebooks can silently go stale vs. the claimed numbers. | Every notebook ends with asserts against CLAIMS.md; `uv run jupyter execute` is part of each gate |

## Results Log

| Date | Stage | Headline number | Notes |
| ---- | ----- | --------------- | ----- |
| 2026-07-14 | 0 | 14 tests green; frozen sync verified in scratch copy | torch pinned 2.13.0 **+cpu** (critic fix: CUDA build violated the ≤10-min volunteer tier); manifest tooling round-trip tested |
| 2026-07-14 | 1 | 14 gate_stage1 tests green; 01_simulators.ipynb executes end-to-end | GBM exact scheme + Heston full truncation Euler; all statistical tolerances 4 SE, derived in docstrings; critic fix: added Heston discounted-price martingale test (exact property of log-Euler step) |
| 2026-07-14 | 2 | 13 gate_stage2 tests green; convergence slope ≈ −0.5; at 50 bps calibrated WW CVaR95 3.74 vs naive delta 5.28 | Critic fixes: Leland calibration grid saturated at its edge twice → log-spaced grid to 316 + permanent interior-argmin guard test; numpy 2.x scalar-conversion fix in pricing; float-epsilon guard in CVaR tail count. Note: Leland beats WW at 5 bps (1.23 vs 1.37) — band advantage only kicks in at higher costs |
| 2026-07-14 | 3 | Frozen TEST table: at 50 bps CVaR95 delta 3.5041 / Leland 2.7287 / WW 2.7940; WW turnover 1.85 vs delta 3.51 | 9 claims registered + verify_claims tooling; TEST guard (PermissionError without final=True); path-set SHA-256s recorded; critic fix: manifest allow_change=False freeze guard so sign-off notebooks cannot silently re-freeze changed benchmarks. Honest note: Leland edges WW on pure CVaR95 at all positive costs |
| 2026-07-14 | 4 | H1 verdict: **not supported** (honest negative on the conjunction). CVaR half held: policy 1.6477±0.0076 vs WW 1.8006 at 20 bps, paired diff −0.1534, CI [−0.1644, −0.1407]; turnover half failed: 2.745 vs 2.271 | 15 policies (3 costs × 5 seeds), float32 training + float64 publication; torch/numpy engine equivalence 5e-14; published-metrics reproduction to 1e-6 green; tolerances.json written. Critic notes: (a) pre-registered turnover condition is not implied by the shared objective — policy rationally buys tail relief with extra trading; (b) at 50 bps the policy converges but loses to calibrated WW on the shared objective (3.786 vs 3.740) — optimizer/architecture headroom, recorded not hidden; (c) Leland still wins pure CVaR95 at 5/20 bps |
