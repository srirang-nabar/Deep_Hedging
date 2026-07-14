"""Reproduction entry point.

Responsibility: `uv run python -m deep_hedging.reproduce --tier N` re-runs
the verification tiers documented in REPRODUCING.md — tier 1 verifies
committed artifacts, tier 2 recomputes tables and figures from committed
weights and seeds, tier 3 retrains from scratch and checks results against
the tolerance bands in results/tolerances.json.
"""
