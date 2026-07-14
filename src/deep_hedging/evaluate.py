"""Hedging engine and Monte Carlo evaluation.

Responsibility: simulate discrete rebalancing of an arbitrary strategy
(callable HedgeState -> target position) with proportional transaction costs
and return the terminal hedging-error (P&L) distribution with an exact
accounting decomposition. Also the risk metrics (CVaR, mean-CVaR objective)
that both baseline calibration (Stage 2) and policy training (Stage 4)
optimize — one objective, shared by everyone, so comparisons are fair.

Scenario: the hedger has SOLD one European call at the Black-Scholes price
(premium received as initial cash) and trades the underlying to manage the
risk. P&L per path = premium + trading gains - transaction costs - payoff;
negative P&L is a loss. With r=0 (the project default) this identity is exact
to machine precision and is enforced by a property test; with r != 0 the cash
account accrues interest and the reported components are un-accrued.

Path-set discipline: three disjoint seeded pools — TRAIN (calibration and
policy training), VAL (model selection and tuning), TEST (final published
numbers only). generate_path_set refuses TEST unless the caller passes
final=True, which only sign-off contexts (results notebooks and published-
metrics verification tests) may do. SHA-256 fingerprints of all three pools
are recorded in results/path_sets.json and tracked by the manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from deep_hedging.pricing import bs_price
from deep_hedging.simulate import GBMParams, simulate_gbm

PATH_SET_SEEDS = {"TRAIN": 101, "VAL": 202, "TEST": 303}
PATH_SET_SIZES = {"TRAIN": 100_000, "VAL": 50_000, "TEST": 100_000}

# The canonical experiment: hedging a 3-month ATM call, ~daily rebalancing,
# under driftless GBM (mu = 0: evaluation under the risk-neutral measure, so
# the BS premium is fair and zero-cost mean P&L is 0 up to MC noise).
CANONICAL_GBM = GBMParams(s0=100.0, mu=0.0, sigma=0.2)
CANONICAL = {"strike": 100.0, "horizon": 0.25, "n_steps": 63}


def generate_path_set(name: str, *, final: bool = False, n_paths: int | None = None) -> np.ndarray:
    """Generate one of the three canonical path pools by its fixed seed.

    TEST requires final=True — the code-level guard against peeking at the
    exam. Note numpy's row-major fill means a smaller n_paths draw is an
    exact prefix of the full pool (Stage 2 calibrated on the first 50k TRAIN
    paths), so partial draws never leak another pool's randomness.
    """
    if name not in PATH_SET_SEEDS:
        raise KeyError(f"unknown path set {name!r}; expected one of {sorted(PATH_SET_SEEDS)}")
    if name == "TEST" and not final:
        raise PermissionError(
            "TEST paths are reserved for final sign-off (results notebooks and "
            "published-metrics verification tests). Pass final=True only there; "
            "use TRAIN or VAL for development."
        )
    n = PATH_SET_SIZES[name] if n_paths is None else n_paths
    return simulate_gbm(
        CANONICAL_GBM,
        n_paths=n,
        n_steps=CANONICAL["n_steps"],
        horizon=CANONICAL["horizon"],
        seed=PATH_SET_SEEDS[name],
    )


def path_set_fingerprint(paths: np.ndarray) -> str:
    """SHA-256 of the raw float64 path bytes — the identity of a path pool.
    Bit-reproducible on any CPU (PCG64 + exact GBM scheme are deterministic)."""
    return hashlib.sha256(np.ascontiguousarray(paths).tobytes()).hexdigest()


# ------------------------------------------------------------- metrics ---


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional value-at-risk: the mean of the worst (1-alpha) fraction of
    losses (losses = -pnl, so larger = worse). CVaR(95%) answers: 'on the
    worst 1-in-20 outcomes, how much do I lose on average?'"""
    losses = np.asarray(losses)
    # -1e-9 guards binary-float artifacts: 0.05 * 100 = 5.000000000000004,
    # which would otherwise ceil to a 6-element tail
    n_tail = max(1, int(np.ceil((1.0 - alpha) * losses.size - 1e-9)))
    tail = np.partition(losses, losses.size - n_tail)[losses.size - n_tail :]
    return float(tail.mean())


def mean_cvar_objective(pnl: np.ndarray, *, alpha: float = 0.95, lam: float = 1.0) -> float:
    """J = mean(loss) + lam * CVaR_alpha(loss), loss = -pnl. Lower is better.
    The SAME objective is used to calibrate classical baselines (Stage 2) and
    to train the learned policy (Stage 4) — the baseline fairness protocol."""
    loss = -np.asarray(pnl)
    return float(loss.mean() + lam * cvar(loss, alpha))


# ------------------------------------------------------------- engine ---


@dataclass(frozen=True)
class HedgeState:
    """Everything a strategy may look at when choosing its next position.
    Arrays are per-path; scalars are shared. Strategies ignore fields they
    don't need."""

    spot: np.ndarray  # current spot, shape (n_paths,)
    holding: np.ndarray  # current position in shares, shape (n_paths,)
    time_to_expiry: float
    step: int
    dt: float
    strike: float
    sigma: float
    r: float
    cost_rate: float


Strategy = Callable[[HedgeState], np.ndarray]


@dataclass(frozen=True)
class HedgeResult:
    """Terminal hedging-error distribution plus its exact decomposition.
    All arrays have shape (n_paths,). pnl = premium + trading_pnl - costs
    - payoff (exact at r=0). turnover = total shares traded, including the
    final liquidation trade."""

    pnl: np.ndarray
    premium: float
    trading_pnl: np.ndarray
    costs: np.ndarray
    payoff: np.ndarray
    turnover: np.ndarray
    holdings: np.ndarray | None = None  # (n_paths, n_steps) if requested


def simulate_hedge(
    paths: np.ndarray,
    *,
    strike: float,
    horizon: float,
    sigma: float,
    strategy: Strategy,
    cost_rate: float = 0.0,
    r: float = 0.0,
    store_holdings: bool = False,
) -> HedgeResult:
    """Play a strategy through pre-simulated spot paths (n_paths, n_steps+1).

    Mechanics per step k: the strategy sees (spot, current holding, time to
    expiry) and names a target position; the engine trades the difference,
    charging cost_rate * |shares traded| * spot. After the last rebalance the
    option settles at max(S_T - K, 0) and the stock position is liquidated
    (paying costs on it too — carrying inventory to expiry is not free).
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = horizon / n_steps

    s0 = float(paths[0, 0])
    premium = float(bs_price(s0, strike, horizon, sigma, r))

    cash = np.full(n_paths, premium)
    holding = np.zeros(n_paths)
    trading_pnl = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    holdings = np.empty((n_paths, n_steps)) if store_holdings else None
    accrual = float(np.exp(r * dt))

    for k in range(n_steps):
        spot_k = paths[:, k]
        state = HedgeState(
            spot=spot_k,
            holding=holding,
            time_to_expiry=horizon - k * dt,
            step=k,
            dt=dt,
            strike=strike,
            sigma=sigma,
            r=r,
            cost_rate=cost_rate,
        )
        target = np.asarray(strategy(state), dtype=np.float64)
        trade = target - holding
        step_cost = cost_rate * np.abs(trade) * spot_k
        cash -= trade * spot_k + step_cost
        costs += step_cost
        turnover += np.abs(trade)
        holding = target
        if store_holdings:
            holdings[:, k] = holding
        trading_pnl += holding * (paths[:, k + 1] - spot_k)
        cash *= accrual

    s_t = paths[:, -1]
    payoff = np.maximum(s_t - strike, 0.0)
    liq_cost = cost_rate * np.abs(holding) * s_t
    cash += holding * s_t - liq_cost
    costs += liq_cost
    turnover += np.abs(holding)

    pnl = cash - payoff
    return HedgeResult(
        pnl=pnl,
        premium=premium,
        trading_pnl=trading_pnl,
        costs=costs,
        payoff=payoff,
        turnover=turnover,
        holdings=holdings,
    )


# ----------------------------------------------- metrics & results table ---


def bootstrap_cvar_ci(
    losses: np.ndarray,
    *,
    alpha: float = 0.95,
    n_boot: int = 1000,
    seed: int = 7,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Seeded bootstrap CI for CVaR: resample paths with replacement n_boot
    times, recompute CVaR each time, take the percentile interval. Fixed
    seed -> identical CI on every run (a determinism test enforces this)."""
    rng = np.random.default_rng(seed)
    n = losses.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = cvar(losses[rng.integers(0, n, n)], alpha)
    lo, hi = np.quantile(stats, [(1 - ci_level) / 2, (1 + ci_level) / 2])
    return float(lo), float(hi)


def summarize_hedge(result: HedgeResult, *, n_boot: int = 1000, boot_seed: int = 7) -> dict:
    """The metric row every strategy is judged by: mean/std of P&L, CVaR(95%)
    and CVaR(99%) of loss with bootstrap CIs, and mean turnover."""
    loss = -result.pnl
    return {
        "mean_pnl": float(result.pnl.mean()),
        "std_pnl": float(result.pnl.std(ddof=1)),
        "cvar95": cvar(loss, 0.95),
        "cvar95_ci": bootstrap_cvar_ci(loss, alpha=0.95, n_boot=n_boot, seed=boot_seed),
        "cvar99": cvar(loss, 0.99),
        "cvar99_ci": bootstrap_cvar_ci(loss, alpha=0.99, n_boot=n_boot, seed=boot_seed + 1),
        "turnover": float(result.turnover.mean()),
    }


def build_baseline_table(
    paths: np.ndarray,
    *,
    strategy_factories: dict[str, Callable[[float], Strategy]],
    cost_levels_bps: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0),
    strike: float = CANONICAL["strike"],
    horizon: float = CANONICAL["horizon"],
    sigma: float = CANONICAL_GBM.sigma,
    n_boot: int = 1000,
    boot_seed: int = 7,
    meta: dict | None = None,
) -> dict:
    """Evaluate every (strategy x cost level) cell on the given paths.

    strategy_factories maps a display name to a factory cost_bps -> Strategy,
    so calibrated parameters can differ per cost level (they do). The caller
    supplies factories to keep this module independent of baselines.py."""
    rows = []
    for bps in cost_levels_bps:
        for name, factory in strategy_factories.items():
            result = simulate_hedge(
                paths,
                strike=strike,
                horizon=horizon,
                sigma=sigma,
                strategy=factory(bps),
                cost_rate=bps / 10_000.0,
            )
            rows.append(
                {"cost_bps": float(bps), "strategy": name}
                | summarize_hedge(result, n_boot=n_boot, boot_seed=boot_seed)
            )
    return {
        "config": {
            "n_paths": int(paths.shape[0]),
            "n_steps": int(paths.shape[1] - 1),
            "strike": strike,
            "horizon": horizon,
            "sigma": sigma,
            "cost_levels_bps": list(cost_levels_bps),
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "path_set_sha256": path_set_fingerprint(paths),
        }
        | (meta or {}),
        "rows": rows,
    }


_TABLE_COLUMNS = [
    ("cost (bps)", lambda r: f"{r['cost_bps']:g}"),
    ("strategy", lambda r: r["strategy"]),
    ("mean P&L", lambda r: f"{r['mean_pnl']:.4f}"),
    ("std", lambda r: f"{r['std_pnl']:.4f}"),
    ("CVaR95", lambda r: f"{r['cvar95']:.4f}"),
    ("CVaR95 95% CI", lambda r: f"[{r['cvar95_ci'][0]:.4f}, {r['cvar95_ci'][1]:.4f}]"),
    ("CVaR99", lambda r: f"{r['cvar99']:.4f}"),
    ("CVaR99 95% CI", lambda r: f"[{r['cvar99_ci'][0]:.4f}, {r['cvar99_ci'][1]:.4f}]"),
    ("turnover", lambda r: f"{r['turnover']:.3f}"),
]


def baseline_table_to_markdown(table: dict) -> str:
    """Render the machine-readable table as the human-readable one. The two
    must agree — a round-trip test parses this text back and compares."""
    header = "| " + " | ".join(name for name, _ in _TABLE_COLUMNS) + " |"
    sep = "|" + "|".join(" --- " for _ in _TABLE_COLUMNS) + "|"
    lines = [header, sep]
    for row in table["rows"]:
        lines.append("| " + " | ".join(fmt(row) for _, fmt in _TABLE_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def parse_baseline_table_markdown(text: str) -> list[dict]:
    """Parse the markdown table back into numeric rows (values at rendered
    precision) — the other half of the json <-> md round trip."""
    rows = []
    lines = [ln for ln in text.strip().splitlines() if ln.startswith("|")]
    for line in lines[2:]:  # skip header + separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]

        def ci(cell):
            lo, hi = cell.strip("[]").split(",")
            return (float(lo), float(hi))

        rows.append(
            {
                "cost_bps": float(cells[0]),
                "strategy": cells[1],
                "mean_pnl": float(cells[2]),
                "std_pnl": float(cells[3]),
                "cvar95": float(cells[4]),
                "cvar95_ci": ci(cells[5]),
                "cvar99": float(cells[6]),
                "cvar99_ci": ci(cells[7]),
                "turnover": float(cells[8]),
            }
        )
    return rows
