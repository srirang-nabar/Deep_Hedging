# Reproducing the results

Everything starts from a frozen environment:

```bash
uv sync --frozen
```

`uv.lock` and `.python-version` are the contract — torch is pinned exactly
(CPU build; no GPU or CUDA download involved anywhere). If `uv sync
--frozen` fails, stop and report it; nothing downstream is expected to work.

There are three tiers of reproduction. Pick the one that matches how much
time you have. Wall-clock times were measured on a commodity Linux x86-64
CPU box (see `results/fresh_machine_run.log` for a full transcript).

## Tier 1 — Verify the claims (≈5–10 min)

```bash
uv run python -m deep_hedging.reproduce --tier 1
```

Verifies the SHA-256 of every committed artifact against
`results/MANIFEST.sha256`, then executes all five notebooks top-to-bottom.
The notebooks load **committed artifacts** — trained weights, frozen tables,
the NIFTY snapshot — and `assert` every number registered in `CLAIMS.md`
(each notebook's final cell calls `verify_claims`, which also fails on
stale or unregistered numbers). Equivalent by hand: run the notebooks in
`notebooks/` and `uv run python -m deep_hedging.manifest verify`.

The full test suite is the same guarantee in pytest form:

```bash
uv run pytest            # all stage gates; 71 tests, ~1-2 min measured
```

## Tier 2 — Recompute the results exactly (≈4 min, measured 196s)

```bash
uv run python -m deep_hedging.reproduce --tier 2
```

Rebuilds the published results from committed inputs and compares against
the frozen artifacts: the baseline table is recomputed from the frozen TEST
path set and calibration file (matches to 1e-9), every committed policy is
re-evaluated on TEST (CVaR95/turnover match stored per-seed values to 1e-6),
and the H2 band metric is recomputed from the weights (matches to 1e-9).
This is **exact** reproduction: the committed artifacts are byte-level
reproducible on any CPU because every random draw is seeded (numpy PCG64)
and all evaluation runs in float64.

## Tier 3 — Full retrain (≈1.5–2 h, CPU)

```bash
uv run python -m deep_hedging.reproduce --tier 3
```

Retrains all 15 policies (3 cost levels × 5 seeds, ~4.5–5.5 min each on the
reference box) from scratch with the documented seeds into a scratch
directory, evaluates them on the frozen TEST set, and checks that the
across-seed mean CVaR95 and turnover land inside the tolerance bands
published in `results/tolerances.json` (mean ± 4 across-seed σ).

**What "reproduces" means here:** floating-point nondeterminism across
hardware, BLAS builds, and thread counts makes bit-exact retraining
impossible, so Tier 3 promises **statistical** reproduction — your retrained
metrics must land inside the published bands. This is different from Tiers
1–2, which are **exact**. The distinction is deliberate and documented; a
volunteer whose retrained numbers differ in the third decimal has NOT failed
to reproduce the result.

## Smoke configs

Every training run has a `--smoke` variant exercising the identical code
path in seconds (small paths/epochs):

```bash
uv run python -m deep_hedging.train train --cost-bps 20 --seed 0 --smoke
```

## Hygiene guarantees you can check yourself

- TRAIN/VAL/TEST path pools regenerate bit-identically from their seeds and
  match the fingerprints in `results/path_sets.json` (a gate test does this).
- TEST access without `final=True` raises `PermissionError` — try it.
- Weight sidecars record the fingerprints of exactly the data each policy
  trained on; a leak test asserts TEST's fingerprint appears nowhere.
- The NIFTY snapshot (`data/nifty_daily.csv`) carries source, date range,
  and SHA-256 in `data/nifty_metadata.json`; no network access is needed to
  reproduce anything.
