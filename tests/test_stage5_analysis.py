"""Stage 5 gate: analysis and stress tests.

Pins: the H2 band metric recomputes from committed weights and matches the
frozen artifact (the metric is computed, never eyeballed); pre-registered
thresholds appear verbatim in HYPOTHESES.md and every hypothesis has a
verdict; the misspecified path sets regenerate bit-identically from their
recorded seeds (Heston + NIFTY stationary block bootstrap); the NIFTY
snapshot's integrity and provenance; ablation artifact reproduction; and the
Stage 5 claims register rows.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import deep_hedging
from deep_hedging.analysis import (
    BAND_COST_LEVELS,
    GBM_TEST_DIFF,
    HESTON_SEED,
    NIFTY_BOOT_SEED,
    band_width,
    bootstrap_nifty_paths,
    learned_no_trade_region,
    load_nifty_log_returns,
    ww_no_trade_region,
)
from deep_hedging.claims import verify_claims
from deep_hedging.evaluate import CANONICAL, CANONICAL_GBM, cvar, generate_path_set, path_set_fingerprint, simulate_hedge
from deep_hedging.policy import PolicyStrategy, load_policy
from deep_hedging.simulate import HestonParams, simulate_heston
from deep_hedging.train import TRAIN_SEEDS, weight_path

pytestmark = pytest.mark.gate_stage5

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
BAND = json.loads((RESULTS / "band_analysis.json").read_text())
MISSPEC = json.loads((RESULTS / "misspecification.json").read_text())


def test_h2_band_metric_reproduces():
    """Recompute the band metric at 50 bps for every seed from committed
    weights; IoU and width must equal the frozen artifact to 1e-9. The band
    claim is computed, not eyeballed."""
    entry = BAND["per_cost"]["50.0"]
    ww_region = ww_no_trade_region(50.0, entry["ww_risk_aversion"])
    assert abs(band_width(ww_region) - entry["ww_width"]) < 1e-9
    for i, seed in enumerate(TRAIN_SEEDS):
        policy, _ = load_policy(weight_path(50.0, seed))
        region = learned_no_trade_region(policy)
        iou = np.logical_and(region, ww_region).sum() / np.logical_or(region, ww_region).sum()
        assert abs(iou - entry["iou_per_seed"][i]) < 1e-9
        assert abs(band_width(region) - entry["width_per_seed"][i]) < 1e-9


def test_h2_verdict_logic_and_preregistration():
    """The stored verdict follows mechanically from the stored evidence and
    the pre-registered thresholds, which appear verbatim in HYPOTHESES.md."""
    iou_pass = all(BAND["per_cost"][str(c)]["iou_mean"] >= 0.5 for c in BAND_COST_LEVELS)
    widths = [BAND["per_cost"][str(c)]["width_mean"] for c in BAND_COST_LEVELS]
    monotone_pass = all(a < b for a, b in zip(widths, widths[1:])) and BAND["seeds_monotone_count"] >= 4
    assert BAND["iou_pass"] == iou_pass
    assert BAND["monotone_pass"] == monotone_pass
    assert BAND["h2_verdict"] == ("supported" if (iou_pass and monotone_pass) else "not supported")
    hyp = (PROJECT_ROOT / "HYPOTHESES.md").read_text()
    h2 = hyp.split("## H2")[1].split("## H3")[0]
    assert "IoU ≥ 0.5" in h2 and "at least 4 of 5 seeds" in h2
    assert "_pending_" not in h2


def test_h3_path_sets_regenerate():
    """Both primary misspecified path sets rebuild bit-identically from the
    recorded seeds — the volunteer's guarantee for the stress tests."""
    heston = simulate_heston(
        HestonParams(), n_paths=MISSPEC["config"]["n_paths"],
        n_steps=CANONICAL["n_steps"], horizon=CANONICAL["horizon"], seed=HESTON_SEED,
    )
    assert path_set_fingerprint(heston.spot) == MISSPEC["sets"]["heston"]["path_fingerprint"]
    nifty = bootstrap_nifty_paths(
        block_length=10, n_paths=MISSPEC["config"]["n_paths"],
        n_steps=CANONICAL["n_steps"], seed=NIFTY_BOOT_SEED,
    )
    assert path_set_fingerprint(nifty) == MISSPEC["sets"]["nifty_block10"]["path_fingerprint"]


def test_h3_verdict_logic():
    primary = [MISSPEC["sets"]["heston"], MISSPEC["sets"]["nifty_block10"]]
    not_inverted = all(s["cvar_diff_ci95"][1] < 0.0 for s in primary)
    shrunk = all(s["cvar_diff_mean"] > GBM_TEST_DIFF for s in primary)
    assert MISSPEC["not_inverted"] == not_inverted
    assert MISSPEC["shrunk"] == shrunk
    assert MISSPEC["h3_verdict"] == ("supported" if (not_inverted and shrunk) else "not supported")
    for s in MISSPEC["sets"].values():
        lo, hi = s["cvar_diff_ci95"]
        assert lo <= s["cvar_diff_mean"] <= hi
    hyp = (PROJECT_ROOT / "HYPOTHESES.md").read_text()
    assert "_pending_" not in hyp, "every hypothesis must carry a verdict at the Stage 5 gate"


def test_gbm_reference_diff_matches_claims():
    """The H3 'shrinks' reference constant in analysis.py must equal the
    registered H1 paired diff — they cannot silently diverge."""
    from deep_hedging.claims import read_claims

    assert abs(GBM_TEST_DIFF - read_claims()["H1-CVAR-DIFF-MEAN"]["value"]) < 1e-9


def test_nifty_snapshot_integrity():
    """Committed data matches its manifest hash and metadata; returns are
    plausibly an equity index (annualized vol 10-30%, >2000 observations)."""
    import hashlib
    from deep_hedging import manifest

    csv_path = PROJECT_ROOT / "data" / "nifty_daily.csv"
    meta = json.loads((PROJECT_ROOT / "data" / "nifty_metadata.json").read_text())
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert digest == meta["sha256"]
    assert manifest.read_manifest()["data/nifty_daily.csv"] == digest
    returns = load_nifty_log_returns()
    assert returns.size > 2000
    assert 0.10 < returns.std() * np.sqrt(252) < 0.30


def test_ablation_reproduces_one_seed():
    """Reload the seed-0 stateless policy, evaluate on TEST, match the frozen
    artifact to 1e-6 — same guarantee as the full policies."""
    abl = json.loads((RESULTS / "ablation.json").read_text())
    paths = generate_path_set("TEST", final=True)
    assert path_set_fingerprint(paths) == abl["path_set_sha256"]
    policy, sidecar = load_policy(weight_path(20.0, 0, prefix="ablation"))
    assert sidecar["mask_inventory"] is True
    res = simulate_hedge(
        paths, strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma,
        strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma, use_inventory=False),
        cost_rate=20.0 / 10_000.0,
    )
    row = next(r for r in abl["per_seed"] if r["seed"] == 0)
    assert abs(cvar(-res.pnl, 0.95) - row["cvar95"]) < 1e-6
    assert abs(res.turnover.mean() - row["turnover"]) < 1e-6


def test_stage5_claims_match_artifacts():
    abl = json.loads((RESULTS / "ablation.json").read_text())
    computed = {
        f"H2-WIDTH-{c:g}": BAND["per_cost"][str(c)]["width_mean"] for c in BAND_COST_LEVELS
    } | {
        f"H2-IOU-{c:g}": BAND["per_cost"][str(c)]["iou_mean"] for c in BAND_COST_LEVELS
    } | {
        "H3-HESTON-CVAR-DIFF": MISSPEC["sets"]["heston"]["cvar_diff_mean"],
        "H3-NIFTY-CVAR-DIFF": MISSPEC["sets"]["nifty_block10"]["cvar_diff_mean"],
        "ABL20-CVAR95-MEAN": abl["across_seeds"]["cvar95"]["mean"],
    }
    summary = verify_claims("05_analysis", computed, atol=1e-4)
    assert summary.startswith("verified 9 ")
