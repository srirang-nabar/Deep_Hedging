"""Stage 2 gate: pricing, hedging engine, classical baselines.

Coding tests: Black-Scholes golden values (Hull), put-call parity as a
property test, the engine's exact accounting identity on random strategies,
and zero-cost/analytic-case equivalences. Statistical tests: the zero-cost
convergence 'interview test' (log-log slope -1/2) and the calibrated-WW vs
naive-delta ordering at 50 bps with a bootstrap CI. Tolerances are derived
where statistical; fixed seeds make every test deterministic.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import deep_hedging
from deep_hedging.baselines import (
    bs_delta_strategy,
    load_calibration,
    make_leland_strategy,
    make_whalley_wilmott_strategy,
)
from deep_hedging.evaluate import PATH_SET_SEEDS, cvar, mean_cvar_objective, simulate_hedge
from deep_hedging.pricing import bs_delta, bs_gamma, bs_price
from deep_hedging.simulate import GBMParams, simulate_gbm

pytestmark = pytest.mark.gate_stage2

PROJECT_ROOT = Path(deep_hedging.__file__).resolve().parents[2]
CALIBRATION_PATH = PROJECT_ROOT / "results" / "baseline_calibration.json"

GBM = GBMParams(s0=100.0, mu=0.0, sigma=0.2)
STRIKE, HORIZON, N_STEPS = 100.0, 0.25, 63


def make_paths(n_paths, seed, n_steps=N_STEPS):
    return simulate_gbm(GBM, n_paths=n_paths, n_steps=n_steps, horizon=HORIZON, seed=seed)


# ------------------------------------------------------------- pricing ---


def test_bs_golden_values():
    """Hull, 'Options, Futures and Other Derivatives': S=42, K=40, r=0.10,
    sigma=0.20, T=0.5 -> call 4.76, put 0.81 (2 dp). Delta N(d1) checked on
    the same classic example."""
    call = bs_price(42.0, 40.0, 0.5, 0.20, 0.10, "call")
    put = bs_price(42.0, 40.0, 0.5, 0.20, 0.10, "put")
    assert abs(call - 4.76) < 5e-3
    assert abs(put - 0.81) < 5e-3
    d1 = (np.log(42 / 40) + (0.10 + 0.02) * 0.5) / (0.20 * np.sqrt(0.5))
    from scipy.special import ndtr

    assert abs(bs_delta(42.0, 40.0, 0.5, 0.20, 0.10) - ndtr(d1)) < 1e-12


def test_bs_expiry_edge_cases():
    assert bs_price(110.0, 100.0, 0.0, 0.2) == 10.0
    assert bs_price(90.0, 100.0, 0.0, 0.2) == 0.0
    assert bs_delta(110.0, 100.0, 0.0, 0.2) == 1.0
    assert bs_delta(90.0, 100.0, 0.0, 0.2) == 0.0
    assert bs_gamma(110.0, 100.0, 0.0, 0.2) == 0.0


@settings(max_examples=200, deadline=None)
@given(
    s=st.floats(10.0, 200.0),
    k=st.floats(10.0, 200.0),
    tau=st.floats(0.05, 3.0),
    sigma=st.floats(0.05, 0.8),
    r=st.floats(0.0, 0.10),
)
def test_put_call_parity_property(s, k, tau, sigma, r):
    """c - p = S - K e^{-r tau} must hold for ANY parameters."""
    c = bs_price(s, k, tau, sigma, r, "call")
    p = bs_price(s, k, tau, sigma, r, "put")
    assert abs((c - p) - (s - k * np.exp(-r * tau))) < 1e-9 * (s + k)


# -------------------------------------------------------------- engine ---


@settings(max_examples=25, deadline=None)
@given(
    holding_level=st.floats(-2.0, 2.0),
    cost_bps=st.floats(0.0, 100.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_accounting_identity_property(holding_level, cost_bps, seed):
    """pnl == premium + trading_pnl - costs - payoff exactly (r=0), for a
    random state-dependent strategy under random costs and paths. The engine
    cannot leak or invent money."""
    paths = make_paths(200, seed, n_steps=16)

    def wobbly(state):
        return holding_level * np.sin(state.spot / 10.0 + state.step)

    res = simulate_hedge(
        paths,
        strike=STRIKE,
        horizon=HORIZON,
        sigma=GBM.sigma,
        strategy=wobbly,
        cost_rate=cost_bps / 10_000.0,
    )
    recomposed = res.premium + res.trading_pnl - res.costs - res.payoff
    np.testing.assert_allclose(res.pnl, recomposed, rtol=0, atol=1e-10)


def test_zero_position_equivalence():
    """Holding nothing: pnl = premium - payoff exactly, zero costs/turnover,
    at any cost rate (no trades -> no fees)."""
    paths = make_paths(500, 7)
    res = simulate_hedge(
        paths,
        strike=STRIKE,
        horizon=HORIZON,
        sigma=GBM.sigma,
        strategy=lambda s: np.zeros_like(s.spot),
        cost_rate=0.005,
    )
    np.testing.assert_allclose(res.pnl, res.premium - res.payoff, atol=1e-12)
    assert np.all(res.costs == 0) and np.all(res.turnover == 0)


def test_static_share_equivalence_zero_cost():
    """Buy-and-hold one share, zero costs: pnl = premium + (S_T - S_0)
    - payoff analytically, per path."""
    paths = make_paths(500, 8)
    res = simulate_hedge(
        paths,
        strike=STRIKE,
        horizon=HORIZON,
        sigma=GBM.sigma,
        strategy=lambda s: np.ones_like(s.spot),
        cost_rate=0.0,
    )
    expected = res.premium + (paths[:, -1] - paths[:, 0]) - res.payoff
    np.testing.assert_allclose(res.pnl, expected, atol=1e-10)


def test_zero_cost_ww_and_leland_reduce_to_delta():
    """With cost_rate = 0 the WW band has zero width and the Leland number is
    zero, so both must produce bit-identical pnl to plain delta hedging."""
    paths = make_paths(1_000, 9)
    kwargs = dict(strike=STRIKE, horizon=HORIZON, sigma=GBM.sigma, cost_rate=0.0)
    base = simulate_hedge(paths, strategy=bs_delta_strategy, **kwargs)
    ww = simulate_hedge(paths, strategy=make_whalley_wilmott_strategy(1.0), **kwargs)
    leland = simulate_hedge(paths, strategy=make_leland_strategy(1.7), **kwargs)
    np.testing.assert_array_equal(base.pnl, ww.pnl)
    np.testing.assert_array_equal(base.pnl, leland.pnl)


def test_cvar_hand_computed():
    """CVaR on a toy distribution: losses 1..100, alpha=0.95 -> mean of the
    worst 5 = 98."""
    losses = np.arange(1.0, 101.0)
    assert cvar(losses, alpha=0.95) == pytest.approx(98.0)


# ---------------------------------------------------------- statistical ---


@pytest.mark.statistical
def test_interview_zero_cost_convergence_slope():
    """With zero costs, discrete BS-delta hedging error shrinks like
    n^{-1/2}: std(pnl) vs rebalance count on a log-log scale is a line of
    slope -0.5. Fit OLS over n in {4,...,512}; assert the slope within
    3 fitted standard errors of -0.5 (and hard bounds [-0.65, -0.35])."""
    freqs = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    stds = []
    for i, n in enumerate(freqs):
        paths = simulate_gbm(GBM, n_paths=20_000, n_steps=int(n), horizon=HORIZON, seed=2000 + i)
        res = simulate_hedge(
            paths, strike=STRIKE, horizon=HORIZON, sigma=GBM.sigma,
            strategy=bs_delta_strategy, cost_rate=0.0,
        )
        stds.append(res.pnl.std(ddof=1))
    x, y = np.log(freqs), np.log(stds)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se_slope = np.sqrt(resid @ resid / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean())))
    assert abs(slope + 0.5) < max(3 * se_slope, 0.02)
    assert -0.65 < slope < -0.35


@pytest.mark.statistical
def test_ordering_calibrated_ww_beats_naive_delta_at_50bps():
    """At 50 bps, calibrated WW must have LOWER CVaR(95%) than naive delta:
    the band avoids most of the cost bleed that naive delta pays. Calibrate
    on TRAIN paths, evaluate on a disjoint path set, and require the 95%
    bootstrap CI (seeded, 500 resamples) of the CVaR difference to exclude 0."""
    cost_rate = 0.005
    train = make_paths(20_000, PATH_SET_SEEDS["TRAIN"])
    gammas = np.logspace(-2, 2, 9)
    objectives = []
    for g in gammas:
        res = simulate_hedge(
            train, strike=STRIKE, horizon=HORIZON, sigma=GBM.sigma,
            strategy=make_whalley_wilmott_strategy(g), cost_rate=cost_rate,
        )
        objectives.append(mean_cvar_objective(res.pnl))
    best_gamma = gammas[int(np.argmin(objectives))]

    eval_paths = make_paths(20_000, 5150)  # disjoint from TRAIN
    common = dict(strike=STRIKE, horizon=HORIZON, sigma=GBM.sigma, cost_rate=cost_rate)
    loss_ww = -simulate_hedge(eval_paths, strategy=make_whalley_wilmott_strategy(best_gamma), **common).pnl
    loss_delta = -simulate_hedge(eval_paths, strategy=bs_delta_strategy, **common).pnl

    rng = np.random.default_rng(99)
    n = loss_ww.size
    diffs = np.empty(500)
    for b in range(500):
        idx = rng.integers(0, n, n)
        diffs[b] = cvar(loss_ww[idx]) - cvar(loss_delta[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    assert hi < 0.0, f"WW-delta CVaR diff CI [{lo:.4f}, {hi:.4f}] must exclude 0"


# ----------------------------------------------------------- artifacts ---


def test_calibration_artifact_exists_and_reproduces():
    """Gate artifact: results/baseline_calibration.json (produced by notebook
    02). Schema-check it, then re-evaluate the chosen WW parameter at 50 bps
    on the recorded seed and assert the recorded objective reproduces to
    1e-9 — the artifact matches the code that claims to have made it."""
    assert CALIBRATION_PATH.exists(), "run notebooks/02_baselines.ipynb to produce it"
    calib = load_calibration(CALIBRATION_PATH)
    assert set(calib["per_cost_level"]) == {"0.0", "5.0", "20.0", "50.0"}

    cfg = calib["config"]
    entry = calib["per_cost_level"]["50.0"]["whalley_wilmott"]
    paths = simulate_gbm(
        GBMParams(**cfg["gbm"]),
        n_paths=cfg["n_paths"], n_steps=cfg["n_steps"],
        horizon=cfg["horizon"], seed=cfg["seed"],
    )
    res = simulate_hedge(
        paths, strike=cfg["strike"], horizon=cfg["horizon"], sigma=cfg["gbm"]["sigma"],
        strategy=make_whalley_wilmott_strategy(entry["risk_aversion"]),
        cost_rate=50.0 / 10_000.0,
    )
    reproduced = mean_cvar_objective(res.pnl, alpha=cfg["alpha"], lam=cfg["lam"])
    assert abs(reproduced - entry["objective"]) < 1e-9


def test_calibrated_params_interior_to_grid():
    """Critic guard: a calibrated parameter sitting on its grid boundary
    means the true optimum is likely outside the grid and the baseline is
    under-tuned — the straw-man failure the fairness protocol must prevent.
    (Zero cost is exempt: all parameters tie there, so the argmin is
    arbitrary.)"""
    calib = load_calibration(CALIBRATION_PATH)
    gamma_grid = calib["grids"]["whalley_wilmott_risk_aversion"]
    leland_grid = calib["grids"]["leland_adjustment_scale"]
    for bps, entry in calib["per_cost_level"].items():
        if float(bps) == 0.0:
            continue
        g = entry["whalley_wilmott"]["risk_aversion"]
        s = entry["leland"]["adjustment_scale"]
        assert gamma_grid[0] < g < gamma_grid[-1], f"WW gamma on boundary at {bps} bps"
        assert leland_grid[0] < s < leland_grid[-1], f"Leland scale on boundary at {bps} bps"


def test_convergence_plot_exists():
    assert (PROJECT_ROOT / "results" / "convergence_check.png").exists(), (
        "run notebooks/02_baselines.ipynb to produce it"
    )
