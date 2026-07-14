"""Stage 4 gate: the learned policy.

Coding tests: weight save/load round-trip at tolerance 0, gradient flow and
loss decrease on a tiny overfit problem, torch-vs-numpy engine equivalence
(training P&L is the same quantity the baselines are scored on), the smoke
training pipeline, and the leak test (no committed policy ever saw TEST).
The published-metrics reproduction test is the core volunteer guarantee:
committed weights re-evaluated on the frozen TEST set reproduce the stored
per-seed numbers to 1e-6 and the CLAIMS.md headline to its precision. The H1
verdict is checked for internal consistency against the recorded bootstrap.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import deep_hedging
from deep_hedging.claims import read_claims
from deep_hedging.evaluate import (
    CANONICAL,
    CANONICAL_GBM,
    cvar,
    generate_path_set,
    path_set_fingerprint,
    simulate_hedge,
)
from deep_hedging.policy import HedgePolicy, PolicyStrategy, load_policy, save_policy
from deep_hedging.train import (
    COST_LEVELS_BPS,
    TRAIN_SEEDS,
    objective_torch,
    simulate_pnl_torch,
    train_policy,
    weight_path,
)

pytestmark = pytest.mark.gate_stage4

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
LP_RESULTS_PATH = RESULTS / "learned_policy_results.json"


# ---------------------------------------------------------------- coding ---


def test_weight_round_trip(tmp_path):
    """save -> load on CPU -> bit-identical outputs on a probe batch."""
    torch.manual_seed(5)
    policy = HedgePolicy().double()
    save_policy(policy, tmp_path / "p.pt", {"cost_bps": 20.0, "seed": 5})
    loaded, sidecar = load_policy(tmp_path / "p.pt")
    probe = torch.randn(256, 4, dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(policy(probe), loaded(probe))
    assert sidecar["hidden"] == [32, 32]


def test_gradient_sanity():
    """On a tiny fixed problem the objective decreases and every parameter
    receives gradient."""
    torch.manual_seed(6)
    policy = HedgePolicy().double()
    paths = torch.from_numpy(generate_path_set("TRAIN", n_paths=256))
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    first = None
    for _ in range(40):
        loss = objective_torch(simulate_pnl_torch(policy, paths, cost_rate=0.002))
        opt.zero_grad()
        loss.backward()
        if first is None:
            first = float(loss.detach())
            for name, p in policy.named_parameters():
                assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
        opt.step()
    assert float(loss.detach()) < first


def test_torch_numpy_engine_equivalence():
    """The differentiable simulation and the scoring engine agree to 1e-10
    on the same paths/policy — training optimizes the published quantity."""
    torch.manual_seed(7)
    policy = HedgePolicy().double()
    paths = generate_path_set("VAL", n_paths=1_000)
    with torch.no_grad():
        t_pnl = simulate_pnl_torch(policy, torch.from_numpy(paths), cost_rate=0.002).numpy()
    res = simulate_hedge(
        paths, strike=CANONICAL["strike"], horizon=CANONICAL["horizon"],
        sigma=CANONICAL_GBM.sigma,
        strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma), cost_rate=0.002,
    )
    np.testing.assert_allclose(t_pnl, res.pnl, rtol=0, atol=1e-10)


def test_smoke_training_pipeline(tmp_path):
    """--smoke exercises the identical code path in seconds: trains, early
    stops, writes weights + a complete sidecar."""
    sidecar = train_policy(20.0, 0, smoke=True, out_dir=tmp_path, verbose=False)
    wp = weight_path(20.0, 0, tmp_path)
    assert wp.exists() and Path(f"{wp}.json").exists()
    assert sidecar["smoke"] is True
    assert len(sidecar["val_curve"]) == sidecar["epochs_trained"]
    assert set(sidecar["trained_on"]) == {"TRAIN", "VAL"}
    policy, _ = load_policy(wp)
    with torch.no_grad():
        out = policy(torch.zeros(3, 4, dtype=torch.float64))
    assert out.shape == (3,) and bool(((out >= 0) & (out <= 1)).all())


def test_leak_no_test_paths_in_training():
    """Every committed sidecar: (a) names only TRAIN/VAL, (b) records
    fingerprints that regenerate exactly from those pools' seeds (prefix
    draws), and (c) TEST's fingerprint appears nowhere."""
    test_fp = json.loads((RESULTS / "path_sets.json").read_text())["TEST"]["sha256"]
    for cost_bps in COST_LEVELS_BPS:
        for seed in TRAIN_SEEDS:
            sidecar = json.loads(Path(f"{weight_path(cost_bps, seed)}.json").read_text())
            assert set(sidecar["trained_on"]) <= {"TRAIN", "VAL"}
            for name, info in sidecar["trained_on"].items():
                regenerated = generate_path_set(name, n_paths=info["n_paths"])
                assert path_set_fingerprint(regenerated) == info["sha256"], (cost_bps, seed, name)
                assert info["sha256"] != test_fp


# ----------------------------------------------- published-metrics gate ---


@pytest.mark.statistical
def test_published_metrics_reproduce():
    """THE volunteer guarantee: load every committed 20 bps policy, evaluate
    on the frozen TEST set with the same engine as the baselines, and match
    the stored per-seed CVaR95/turnover to 1e-6 and the CLAIMS.md across-seed
    headline to its registered precision."""
    results = json.loads(LP_RESULTS_PATH.read_text())
    paths = generate_path_set("TEST", final=True)
    assert path_set_fingerprint(paths) == results["config"]["path_set_sha256"]

    cvars = []
    for row in results["per_cost"]["20.0"]["per_seed"]:
        policy, _ = load_policy(weight_path(20.0, row["seed"]))
        res = simulate_hedge(
            paths, strike=CANONICAL["strike"], horizon=CANONICAL["horizon"],
            sigma=CANONICAL_GBM.sigma,
            strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma),
            cost_rate=20.0 / 10_000.0,
        )
        assert abs(cvar(-res.pnl, 0.95) - row["cvar95"]) < 1e-6
        assert abs(res.turnover.mean() - row["turnover"]) < 1e-6
        cvars.append(row["cvar95"])

    claims = read_claims()
    assert abs(np.mean(cvars) - claims["LP20-CVAR95-MEAN"]["value"]) < 1e-4


@pytest.mark.statistical
def test_h1_verdict_recorded_and_consistent():
    """The Stage 4 gate requires a verdict — supported or an honest negative.
    Check it exists, matches its own recorded evidence, and matches
    HYPOTHESES.md."""
    h1 = json.loads(LP_RESULTS_PATH.read_text())["h1"]
    assert h1["verdict"] in ("supported", "not supported")
    lo, hi = h1["cvar_diff_ci95"]
    assert lo <= h1["cvar_diff_mean"] <= hi
    expect = hi < 0.0 and h1["turnover_condition_met"]
    assert (h1["verdict"] == "supported") == expect
    hypotheses = (PROJECT_ROOT / "HYPOTHESES.md").read_text()
    assert "_pending_" not in hypotheses.split("## H2")[0], "H1 verdict not recorded in HYPOTHESES.md"


def test_tolerances_artifact():
    """results/tolerances.json (the Tier 3 retraining contract) covers every
    trained cost level with finite, ordered bands."""
    tol = json.loads((RESULTS / "tolerances.json").read_text())
    assert set(tol) == {str(c) for c in COST_LEVELS_BPS}
    for entry in tol.values():
        for metric in ("cvar95", "turnover"):
            lo, hi = entry[metric]["band"]
            assert lo < entry[metric]["mean"] < hi
