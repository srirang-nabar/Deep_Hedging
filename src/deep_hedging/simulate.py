"""Market path simulators.

Responsibility: generate seeded, vectorized price paths (n_paths x n_steps+1)
for the models used in this project — geometric Brownian motion (exact
log-normal scheme) and Heston stochastic volatility (full truncation Euler,
Lord et al. 2010). Every simulator takes an explicit seed or numpy Generator;
nothing here touches global random state. Simulation runs in float64 and is
cast to the requested dtype at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_generator(seed: int | np.random.Generator) -> np.random.Generator:
    """Explicit-seed policy: an int seeds a fresh Generator, a Generator is
    used as-is (and its state advances — two simulators sharing one Generator
    draw from the same stream sequentially)."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class GBMParams:
    """dS_t = mu S_t dt + sigma S_t dW_t."""

    s0: float = 100.0
    mu: float = 0.0
    sigma: float = 0.2


def simulate_gbm(
    params: GBMParams,
    *,
    n_paths: int,
    n_steps: int,
    horizon: float,
    seed: int | np.random.Generator,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Simulate GBM paths on an equally spaced grid over [0, horizon].

    Uses the exact solution S_{t+dt} = S_t exp((mu - sigma^2/2) dt
    + sigma sqrt(dt) Z), so the terminal distribution is exactly log-normal
    at any step count — no discretization bias.

    Returns an array of shape (n_paths, n_steps + 1); column 0 is s0.
    """
    rng = _as_generator(seed)
    dt = horizon / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (params.mu - 0.5 * params.sigma**2) * dt + params.sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = params.s0
    paths[:, 1:] = params.s0 * np.exp(log_paths)
    return paths.astype(dtype, copy=False)


@dataclass(frozen=True)
class HestonParams:
    """dS_t = mu S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW^v_t,
    d<W^S, W^v>_t = rho dt.

    Literature defaults: kappa=2, theta=0.04, sigma_v=0.3, rho=-0.7. These
    satisfy the Feller condition 2 kappa theta > sigma_v^2 (0.16 > 0.09).
    """

    s0: float = 100.0
    v0: float = 0.04
    mu: float = 0.0
    kappa: float = 2.0
    theta: float = 0.04
    sigma_v: float = 0.3
    rho: float = -0.7


@dataclass(frozen=True)
class HestonPaths:
    """spot and variance are both (n_paths, n_steps + 1). variance is the
    effective (truncated, hence non-negative) variance the spot dynamics
    actually used at each step."""

    spot: np.ndarray
    variance: np.ndarray


def simulate_heston(
    params: HestonParams,
    *,
    n_paths: int,
    n_steps: int,
    horizon: float,
    seed: int | np.random.Generator,
    dtype: np.dtype | type = np.float64,
) -> HestonPaths:
    """Simulate Heston paths with the full truncation Euler scheme.

    The raw variance process may go negative; full truncation propagates the
    raw value but plugs v^+ = max(v, 0) into every coefficient (Lord et al.
    2010, the scheme with the smallest bias among Euler fixes). The spot uses
    a log-Euler step, so spot prices are positive by construction. The
    returned variance is the truncated v^+ path.
    """
    rng = _as_generator(seed)
    dt = horizon / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_perp = np.sqrt(1.0 - params.rho**2)

    log_spot = np.full(n_paths, np.log(params.s0))
    v_raw = np.full(n_paths, float(params.v0))

    spot = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variance = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    spot[:, 0] = params.s0
    variance[:, 0] = max(params.v0, 0.0)

    for k in range(n_steps):
        v_plus = np.maximum(v_raw, 0.0)
        z_s = rng.standard_normal(n_paths)
        z_perp = rng.standard_normal(n_paths)
        z_v = params.rho * z_s + rho_perp * z_perp

        log_spot = log_spot + (params.mu - 0.5 * v_plus) * dt + np.sqrt(v_plus) * sqrt_dt * z_s
        v_raw = (
            v_raw
            + params.kappa * (params.theta - v_plus) * dt
            + params.sigma_v * np.sqrt(v_plus) * sqrt_dt * z_v
        )

        spot[:, k + 1] = np.exp(log_spot)
        variance[:, k + 1] = np.maximum(v_raw, 0.0)

    return HestonPaths(
        spot=spot.astype(dtype, copy=False),
        variance=variance.astype(dtype, copy=False),
    )
