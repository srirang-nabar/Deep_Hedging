# Reproducing the results

Everything starts from a frozen environment:

```bash
uv sync --frozen
```

`uv.lock` and `.python-version` are the contract — torch is pinned exactly.
If `uv sync --frozen` fails, stop and report it; nothing downstream is
expected to work.

There are three tiers of reproduction. Pick the one that matches how much
time you have. (Sections marked *(filled in at stage N)* grow as the project
advances; measured wall-clock times land here in Stage 6.)

## Tier 1 — Verify the claims (≤10 min, CPU laptop)

Run the numbered notebooks in `notebooks/` top to bottom (or
`uv run jupyter execute notebooks/*.ipynb`). They load **committed
artifacts** — trained weights, frozen tables, data snapshots — and `assert`
every headline number against `CLAIMS.md`, not just print it. Then check
artifact integrity:

```bash
uv run python -m deep_hedging.manifest verify
```

*(notebook list filled in as stages complete)*

## Tier 2 — Recompute the results (≤1 hr, CPU)

Re-run evaluation from committed weights and the frozen path-set seeds;
regenerate every table and figure from scratch:

```bash
uv run python -m deep_hedging.reproduce --tier 2
```

*(available from Stage 4)*

## Tier 3 — Full retrain (hours, CPU-feasible)

Retrain the policies from scratch with the documented seeds:

```bash
uv run python -m deep_hedging.reproduce --tier 3
```

**What "reproduces" means here:** floating-point nondeterminism across
hardware and BLAS builds makes bit-exact retraining impossible, so Tier 3
promises **statistical** reproduction — your retrained metrics must land
inside the across-seed tolerance bands published in
`results/tolerances.json`. This is different from Tiers 1–2, which are
**exact**: committed weights re-evaluated on the frozen paths must hit the
published numbers to numerical tolerance on any CPU.

*(available from Stage 4)*
