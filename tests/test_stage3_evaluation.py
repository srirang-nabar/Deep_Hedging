"""Stage 3 gate: evaluation harness and frozen test sets.

The frozen baseline table is the benchmark every later stage compares
against, so these tests pin: the TEST access guard, path-set identity
(regenerated fingerprints match the recorded ones and are pairwise
distinct), metric correctness on hand-computable toys, bootstrap
determinism, the json <-> md round trip, the zero-cost coincidence harness
check, and that CLAIMS.md rows equal the frozen artifact exactly.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import deep_hedging
from deep_hedging.claims import read_claims, verify_claims
from deep_hedging.evaluate import (
    PATH_SET_SEEDS,
    PATH_SET_SIZES,
    HedgeResult,
    baseline_table_to_markdown,
    bootstrap_cvar_ci,
    cvar,
    generate_path_set,
    parse_baseline_table_markdown,
    path_set_fingerprint,
    summarize_hedge,
)

pytestmark = pytest.mark.gate_stage3

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
TABLE_PATH = RESULTS / "baseline_table.json"
PATH_SETS_PATH = RESULTS / "path_sets.json"


def toy_result():
    pnl = np.array([1.0, -1.0, 2.0, -2.0, 0.0, -4.0, 3.0, -3.0, 0.5, -0.5])
    return HedgeResult(
        pnl=pnl,
        premium=1.0,
        trading_pnl=pnl,  # unused by metrics
        costs=np.zeros_like(pnl),
        payoff=np.zeros_like(pnl),
        turnover=np.full_like(pnl, 2.5),
    )


# ---------------------------------------------------------------- metrics ---


def test_metrics_hand_computed():
    """mean/std/cvar/turnover on a 10-value toy: CVaR95 tail = 1 worst loss
    (4.0), CVaR99 likewise; turnover constant 2.5."""
    m = summarize_hedge(toy_result(), n_boot=10)
    assert m["mean_pnl"] == pytest.approx(-0.4)
    assert m["std_pnl"] == pytest.approx(np.std(toy_result().pnl, ddof=1))
    assert m["cvar95"] == pytest.approx(4.0)
    assert m["cvar99"] == pytest.approx(4.0)
    assert m["turnover"] == pytest.approx(2.5)


def test_bootstrap_determinism():
    """Same boot seed -> bit-identical CI; different seed -> different CI."""
    rng = np.random.default_rng(3)
    losses = rng.standard_normal(5_000)
    a = bootstrap_cvar_ci(losses, n_boot=200, seed=11)
    b = bootstrap_cvar_ci(losses, n_boot=200, seed=11)
    c = bootstrap_cvar_ci(losses, n_boot=200, seed=12)
    assert a == b
    assert a != c


def test_table_round_trip():
    """json -> markdown -> parse recovers every number at rendered
    precision (4 dp metrics, 3 dp turnover)."""
    table = json.loads(TABLE_PATH.read_text())
    parsed = parse_baseline_table_markdown(baseline_table_to_markdown(table))
    assert len(parsed) == len(table["rows"])
    for got, want in zip(parsed, table["rows"]):
        assert got["strategy"] == want["strategy"]
        assert got["cost_bps"] == want["cost_bps"]
        for key, tol in [("mean_pnl", 5e-5), ("std_pnl", 5e-5), ("cvar95", 5e-5),
                         ("cvar99", 5e-5), ("turnover", 5e-4)]:
            assert abs(got[key] - want[key]) <= tol, (key, got, want)
        for key in ("cvar95_ci", "cvar99_ci"):
            assert np.allclose(got[key], want[key], atol=5e-5)


# -------------------------------------------------------------- path sets ---


def test_test_paths_guarded():
    """The exam stays sealed: TEST without final=True raises."""
    with pytest.raises(PermissionError):
        generate_path_set("TEST", n_paths=10)
    assert generate_path_set("TEST", n_paths=10, final=True).shape == (10, 64)
    with pytest.raises(KeyError):
        generate_path_set("HOLDOUT", n_paths=10)


def test_path_set_fingerprints_recorded_and_distinct():
    """Regenerate all three pools and compare SHA-256 fingerprints to
    results/path_sets.json: pairwise distinct (disjointness) and equal to
    the recorded ones (the volunteer's regeneration guarantee)."""
    recorded = json.loads(PATH_SETS_PATH.read_text())
    fingerprints = {}
    for name in ("TRAIN", "VAL", "TEST"):
        paths = generate_path_set(name, final=(name == "TEST"))
        assert paths.shape == (PATH_SET_SIZES[name], 64)
        fingerprints[name] = path_set_fingerprint(paths)
        assert fingerprints[name] == recorded[name]["sha256"], name
        assert recorded[name]["seed"] == PATH_SET_SEEDS[name]
    assert len(set(fingerprints.values())) == 3


# ------------------------------------------------------------ frozen table ---


def test_zero_cost_strategies_coincide():
    """Harness check: at zero cost every baseline collapses to plain delta
    hedging, so their frozen rows must be IDENTICAL (stronger than the
    statistical indistinguishability the plan asks for), and mean P&L must
    be 0 within 4 SE (fair premium under the risk-neutral measure)."""
    table = json.loads(TABLE_PATH.read_text())
    zero_rows = [r for r in table["rows"] if r["cost_bps"] == 0.0]
    assert len(zero_rows) == 3
    first = zero_rows[0]
    for row in zero_rows[1:]:
        for key in ("mean_pnl", "std_pnl", "cvar95", "cvar99", "turnover"):
            assert row[key] == first[key], key
    se = first["std_pnl"] / np.sqrt(table["config"]["n_paths"])
    assert abs(first["mean_pnl"]) < 4 * se


def test_frozen_table_cell_reproduces():
    """Recompute one full cell (calibrated WW at 50 bps on TEST) from scratch
    and match the frozen artifact to 1e-9 — the table is what the code says
    it is, not a hand-edited file."""
    from deep_hedging.baselines import load_calibration, make_whalley_wilmott_strategy
    from deep_hedging.evaluate import simulate_hedge

    table = json.loads(TABLE_PATH.read_text())
    row = next(
        r for r in table["rows"]
        if r["cost_bps"] == 50.0 and r["strategy"] == "whalley_wilmott"
    )
    calib = load_calibration(RESULTS / "baseline_calibration.json")
    gamma = calib["per_cost_level"]["50.0"]["whalley_wilmott"]["risk_aversion"]
    cfg = table["config"]
    paths = generate_path_set("TEST", final=True)
    assert path_set_fingerprint(paths) == cfg["path_set_sha256"]
    res = simulate_hedge(
        paths, strike=cfg["strike"], horizon=cfg["horizon"], sigma=cfg["sigma"],
        strategy=make_whalley_wilmott_strategy(gamma), cost_rate=50.0 / 10_000.0,
    )
    loss = -res.pnl
    assert abs(cvar(loss, 0.95) - row["cvar95"]) < 1e-9
    assert abs(cvar(loss, 0.99) - row["cvar99"]) < 1e-9
    assert abs(res.pnl.mean() - row["mean_pnl"]) < 1e-9
    assert abs(res.turnover.mean() - row["turnover"]) < 1e-9


def test_claims_match_frozen_table():
    """Every Stage 3 row in CLAIMS.md equals the frozen table value at the
    register's precision — via the same verify_claims used by notebook 03."""
    table = json.loads(TABLE_PATH.read_text())
    tag = {"bs_delta": "DELTA", "leland": "LELAND", "whalley_wilmott": "WW"}
    computed = {}
    for row in table["rows"]:
        if row["cost_bps"] in (20.0, 50.0):
            computed[f"BT{int(row['cost_bps'])}-{tag[row['strategy']]}-CVAR95"] = row["cvar95"]
        if row["cost_bps"] == 50.0:
            computed[f"BT50-{tag[row['strategy']]}-TURNOVER"] = row["turnover"]
    summary = verify_claims("03_baseline_results", computed, atol=1e-4)
    assert summary.startswith("verified 9 ")
    assert all(v["stage"] == "3" for k, v in read_claims().items() if k.startswith("BT"))
