"""Classical hedging baselines.

Responsibility: the three classical strategies the learned policy must beat —
Black-Scholes delta, Leland-adjusted delta, and the Whalley-Wilmott no-trade
band — plus the calibration routine that grid-searches their free parameters
per cost level on the same mean-CVaR objective and the same TRAIN paths the
learned policy will use (baseline fairness protocol). Published comparisons
are against calibrated baselines only; the calibration grid and chosen
parameters are saved to results/baseline_calibration.json.

All strategies are HedgeState -> target-position callables (the engine's
plug-in interface), so classical rules and the learned policy are scored by
the identical engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deep_hedging.evaluate import (
    PATH_SET_SEEDS,
    HedgeState,
    Strategy,
    mean_cvar_objective,
    simulate_hedge,
)
from deep_hedging.pricing import bs_delta, bs_gamma
from deep_hedging.simulate import GBMParams, simulate_gbm


def bs_delta_strategy(state: HedgeState) -> np.ndarray:
    """Textbook baseline: always rebalance to the Black-Scholes delta.
    Optimal risk control with zero costs; bleeds to death under costs."""
    return bs_delta(state.spot, state.strike, state.time_to_expiry, state.sigma, state.r)


def make_leland_strategy(adjustment_scale: float = 1.0) -> Strategy:
    """Leland (1985): delta-hedge with volatility inflated by the Leland
    number Le = sqrt(2/pi) * cost_rate / (sigma * sqrt(dt)), i.e.
    sigma_adj^2 = sigma^2 (1 + scale * Le). For a short-option hedger the
    higher vol flattens the delta profile, so targets move less between
    steps and turnover drops. scale=1 is the textbook value; the fairness
    protocol calibrates it per cost level."""

    def strategy(state: HedgeState) -> np.ndarray:
        leland_number = (
            np.sqrt(2.0 / np.pi) * state.cost_rate / (state.sigma * np.sqrt(state.dt))
        )
        sigma_adj = state.sigma * np.sqrt(1.0 + adjustment_scale * leland_number)
        return bs_delta(state.spot, state.strike, state.time_to_expiry, sigma_adj, state.r)

    return strategy


def make_whalley_wilmott_strategy(risk_aversion: float) -> Strategy:
    """Whalley-Wilmott (1997) asymptotic no-trade band: half-width
    H = (1.5 * cost_rate * spot * gamma_bs^2 / risk_aversion)^(1/3) around
    the BS delta. Hold while |holding - delta| <= H; otherwise trade to the
    nearest band edge (implemented as a clip). risk_aversion is the free
    parameter the fairness protocol calibrates."""

    def strategy(state: HedgeState) -> np.ndarray:
        delta = bs_delta(state.spot, state.strike, state.time_to_expiry, state.sigma, state.r)
        gamma = bs_gamma(state.spot, state.strike, state.time_to_expiry, state.sigma, state.r)
        half_width = (
            1.5 * state.cost_rate * state.spot * gamma**2 / risk_aversion
        ) ** (1.0 / 3.0)
        return np.clip(state.holding, delta - half_width, delta + half_width)

    return strategy


# --------------------------------------------------------- calibration ---

GAMMA_GRID = np.logspace(-2, 2, 17)  # WW risk aversion
# 0 recovers plain delta. Log-spaced: sigma_adj grows like sqrt(scale), so a
# linear grid wastes resolution at high cost and saturates at low cost
# (critic fix: linear 0..3 and 0..10 grids both hit their upper edge)
LELAND_SCALE_GRID = np.concatenate([[0.0], np.logspace(-1, 2.5, 40)])


def calibrate_baselines(
    *,
    cost_levels_bps: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0),
    gbm: GBMParams = GBMParams(s0=100.0, mu=0.0, sigma=0.2),
    strike: float = 100.0,
    horizon: float = 0.25,
    n_steps: int = 63,
    n_paths: int = 50_000,
    seed: int = PATH_SET_SEEDS["TRAIN"],
    alpha: float = 0.95,
    lam: float = 1.0,
) -> dict:
    """Grid-search each baseline's free parameter per cost level, minimizing
    the shared mean-CVaR objective on TRAIN paths. Returns a JSON-ready dict
    recording config, grids, per-point objectives, and the chosen parameters
    — enough for anyone to re-derive the argmin."""
    paths = simulate_gbm(gbm, n_paths=n_paths, n_steps=n_steps, horizon=horizon, seed=seed)

    def objective(strategy: Strategy, cost_rate: float) -> float:
        result = simulate_hedge(
            paths,
            strike=strike,
            horizon=horizon,
            sigma=gbm.sigma,
            strategy=strategy,
            cost_rate=cost_rate,
        )
        return mean_cvar_objective(result.pnl, alpha=alpha, lam=lam)

    calibration: dict = {
        "config": {
            "gbm": vars(gbm),
            "strike": strike,
            "horizon": horizon,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "seed": seed,
            "path_set": "TRAIN",
            "alpha": alpha,
            "lam": lam,
        },
        "grids": {
            "whalley_wilmott_risk_aversion": GAMMA_GRID.tolist(),
            "leland_adjustment_scale": LELAND_SCALE_GRID.tolist(),
        },
        "cost_levels_bps": list(cost_levels_bps),
        "per_cost_level": {},
    }

    for bps in cost_levels_bps:
        cost_rate = bps / 10_000.0
        ww_objectives = [
            objective(make_whalley_wilmott_strategy(g), cost_rate) for g in GAMMA_GRID
        ]
        leland_objectives = [
            objective(make_leland_strategy(s), cost_rate) for s in LELAND_SCALE_GRID
        ]
        calibration["per_cost_level"][str(bps)] = {
            "whalley_wilmott": {
                "risk_aversion": float(GAMMA_GRID[int(np.argmin(ww_objectives))]),
                "objective": float(np.min(ww_objectives)),
                "objectives": [float(v) for v in ww_objectives],
            },
            "leland": {
                "adjustment_scale": float(LELAND_SCALE_GRID[int(np.argmin(leland_objectives))]),
                "objective": float(np.min(leland_objectives)),
                "objectives": [float(v) for v in leland_objectives],
            },
            "bs_delta": {"objective": objective(bs_delta_strategy, cost_rate)},
        }
    return calibration


def save_calibration(calibration: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")


def load_calibration(path: Path) -> dict:
    return json.loads(path.read_text())
