"""Stage 1 gate: market simulators.

Coding tests pin the reproducibility contract (seed determinism, shape/dtype,
generator isolation). Statistical tests check the simulated distributions
against closed form; every tolerance is derived from a Monte Carlo standard
error (documented per test), never a magic number. All statistical tests use
fixed seeds so they are deterministic in CI.
"""

import numpy as np
import pytest
from scipy import stats

from deep_hedging.simulate import (
    GBMParams,
    HestonParams,
    simulate_gbm,
    simulate_heston,
)

pytestmark = pytest.mark.gate_stage1

GBM = GBMParams(s0=100.0, mu=0.05, sigma=0.2)
HESTON = HestonParams()  # literature defaults


# ---------------------------------------------------------------- coding ---


def test_gbm_seed_determinism():
    """Same seed -> bit-identical paths. This is the volunteer's guarantee."""
    a = simulate_gbm(GBM, n_paths=500, n_steps=32, horizon=1.0, seed=42)
    b = simulate_gbm(GBM, n_paths=500, n_steps=32, horizon=1.0, seed=42)
    assert np.array_equal(a, b)
    c = simulate_gbm(GBM, n_paths=500, n_steps=32, horizon=1.0, seed=43)
    assert not np.array_equal(a, c)


def test_heston_seed_determinism():
    a = simulate_heston(HESTON, n_paths=500, n_steps=32, horizon=1.0, seed=42)
    b = simulate_heston(HESTON, n_paths=500, n_steps=32, horizon=1.0, seed=42)
    assert np.array_equal(a.spot, b.spot)
    assert np.array_equal(a.variance, b.variance)
    c = simulate_heston(HESTON, n_paths=500, n_steps=32, horizon=1.0, seed=43)
    assert not np.array_equal(a.spot, c.spot)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_shape_and_dtype_contracts(dtype):
    n_paths, n_steps = 64, 16
    g = simulate_gbm(GBM, n_paths=n_paths, n_steps=n_steps, horizon=0.5, seed=7, dtype=dtype)
    assert g.shape == (n_paths, n_steps + 1)
    assert g.dtype == dtype
    assert np.all(g[:, 0] == dtype(GBM.s0))
    assert np.all(np.isfinite(g)) and np.all(g > 0)

    h = simulate_heston(HESTON, n_paths=n_paths, n_steps=n_steps, horizon=0.5, seed=7, dtype=dtype)
    assert h.spot.shape == (n_paths, n_steps + 1)
    assert h.variance.shape == (n_paths, n_steps + 1)
    assert h.spot.dtype == dtype and h.variance.dtype == dtype
    assert np.all(h.spot[:, 0] == dtype(HESTON.s0))
    assert np.all(np.isfinite(h.spot)) and np.all(h.spot > 0)


def test_generator_isolation():
    """A simulation seeded with its own int/Generator is unaffected by other
    simulators running in between with different generators."""
    baseline = simulate_gbm(GBM, n_paths=100, n_steps=8, horizon=1.0, seed=123)
    simulate_heston(HESTON, n_paths=100, n_steps=8, horizon=1.0, seed=np.random.default_rng(456))
    again = simulate_gbm(GBM, n_paths=100, n_steps=8, horizon=1.0, seed=123)
    assert np.array_equal(baseline, again)


def test_shared_generator_advances():
    """Two calls on one Generator draw sequentially from the same stream —
    the documented behavior for callers who pass a Generator object."""
    rng = np.random.default_rng(9)
    a = simulate_gbm(GBM, n_paths=50, n_steps=8, horizon=1.0, seed=rng)
    b = simulate_gbm(GBM, n_paths=50, n_steps=8, horizon=1.0, seed=rng)
    assert not np.array_equal(a, b)


# ----------------------------------------------------------- statistical ---

N_BIG = 100_000


@pytest.mark.statistical
def test_gbm_terminal_mean_within_4se():
    """E[S_T] = s0 exp(mu T). Tolerance: 4 sample standard errors of the
    mean at n=100k (SE = sample std / sqrt(n)); false-alarm probability
    ~6e-5 under the null, and the fixed seed makes the outcome deterministic."""
    T = 1.0
    s_t = simulate_gbm(GBM, n_paths=N_BIG, n_steps=8, horizon=T, seed=1001)[:, -1]
    closed_form = GBM.s0 * np.exp(GBM.mu * T)
    se = s_t.std(ddof=1) / np.sqrt(N_BIG)
    assert abs(s_t.mean() - closed_form) < 4 * se


@pytest.mark.statistical
def test_gbm_terminal_variance_within_4se():
    """Var[S_T] = s0^2 e^{2 mu T}(e^{sigma^2 T} - 1). Tolerance: 4 standard
    errors of the sample variance, SE^2 = (m4 - m2^2)/n with m4, m2 the
    empirical central moments (delta-method SE, no normality assumed)."""
    T = 1.0
    s_t = simulate_gbm(GBM, n_paths=N_BIG, n_steps=8, horizon=T, seed=1002)[:, -1]
    closed_form = GBM.s0**2 * np.exp(2 * GBM.mu * T) * (np.exp(GBM.sigma**2 * T) - 1)
    centered = s_t - s_t.mean()
    m2 = np.mean(centered**2)
    m4 = np.mean(centered**4)
    se_var = np.sqrt((m4 - m2**2) / N_BIG)
    assert abs(s_t.var(ddof=1) - closed_form) < 4 * se_var


@pytest.mark.statistical
def test_gbm_log_returns_exactly_normal():
    """One-step log-returns are N((mu - sigma^2/2) dt, sigma^2 dt) exactly
    under the exact scheme. KS test against that fully specified normal at
    the pre-set n=10_000; accept at p > 0.01 (fixed seed -> deterministic)."""
    n, T, n_steps = 10_000, 1.0, 1
    paths = simulate_gbm(GBM, n_paths=n, n_steps=n_steps, horizon=T, seed=1003)
    log_ret = np.log(paths[:, 1] / paths[:, 0])
    dt = T / n_steps
    loc = (GBM.mu - 0.5 * GBM.sigma**2) * dt
    scale = GBM.sigma * np.sqrt(dt)
    result = stats.kstest((log_ret - loc) / scale, "norm")
    assert result.pvalue > 0.01


@pytest.mark.statistical
def test_discounted_price_martingale_under_rn_measure():
    """Under the risk-neutral measure (mu = r), e^{-rt} S_t is a martingale:
    E[e^{-rt} S_t] = s0 at every t. Checked at mid-horizon and terminal
    dates, each within 4 SE of s0 at n=100k."""
    r, T, n_steps = 0.03, 1.0, 8
    rn = GBMParams(s0=100.0, mu=r, sigma=0.2)
    paths = simulate_gbm(rn, n_paths=N_BIG, n_steps=n_steps, horizon=T, seed=1004)
    for k in (n_steps // 2, n_steps):
        t = T * k / n_steps
        discounted = np.exp(-r * t) * paths[:, k]
        se = discounted.std(ddof=1) / np.sqrt(N_BIG)
        assert abs(discounted.mean() - rn.s0) < 4 * se


@pytest.mark.statistical
def test_heston_discounted_price_martingale():
    """The log-Euler spot step makes e^{-rt} S_t an *exact* martingale under
    mu = r: conditionally on v_k, the log-increment is normal with mean
    (r - v_k/2) dt and variance v_k dt, so E[S_{k+1} | F_k] = S_k e^{r dt}
    with no discretization bias. Tolerance: pure MC error, 4 SE at n=100k."""
    r, T, n_steps = 0.03, 1.0, 50
    rn = HestonParams(mu=r)
    h = simulate_heston(rn, n_paths=N_BIG, n_steps=n_steps, horizon=T, seed=1008)
    for k in (n_steps // 2, n_steps):
        t = T * k / n_steps
        discounted = np.exp(-r * t) * h.spot[:, k]
        se = discounted.std(ddof=1) / np.sqrt(N_BIG)
        assert abs(discounted.mean() - rn.s0) < 4 * se


@pytest.mark.statistical
def test_heston_variance_non_negative():
    """Full truncation returns the effective variance v^+ >= 0 everywhere."""
    h = simulate_heston(HESTON, n_paths=20_000, n_steps=250, horizon=1.0, seed=1005)
    assert np.all(h.variance >= 0)


@pytest.mark.statistical
def test_heston_long_run_variance_mean():
    """E[v_T] = theta + (v0 - theta) e^{-kappa T} exactly (CIR mean). Start
    off the long-run level (v0 = 0.09 != theta) so the test has power.
    Tolerance: 4 sample SEs plus a discretization allowance of 1% of theta
    for the Euler bias at dt = 1/500."""
    params = HestonParams(v0=0.09)
    T, n_steps = 3.0, 1500
    h = simulate_heston(params, n_paths=20_000, n_steps=n_steps, horizon=T, seed=1006)
    v_t = h.variance[:, -1]
    expected = params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * T)
    se = v_t.std(ddof=1) / np.sqrt(v_t.shape[0])
    assert abs(v_t.mean() - expected) < 4 * se + 0.01 * params.theta


@pytest.mark.statistical
def test_heston_leverage_effect_sign():
    """corr(d log S, dv) pooled over paths and steps should be ~rho = -0.7
    (drift terms perturb it only at O(dt)). Assert < -0.6: the sampling SE
    at ~2.5M pooled increments is ~(1-rho^2)/sqrt(n) < 1e-3, so the 0.1
    margin is >100 SEs — this fails only if the correlation structure is
    actually wrong, not by chance."""
    h = simulate_heston(HESTON, n_paths=10_000, n_steps=250, horizon=1.0, seed=1007)
    d_log_s = np.diff(np.log(h.spot), axis=1).ravel()
    d_v = np.diff(h.variance, axis=1).ravel()
    corr = np.corrcoef(d_log_s, d_v)[0, 1]
    assert corr < -0.6
