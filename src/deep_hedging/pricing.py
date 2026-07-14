"""Option pricing and greeks.

Responsibility: closed-form Black-Scholes price, delta, and gamma, vectorized
over any broadcastable combination of spot, strike, time to expiry. Used both
as a hedging baseline and to mark the option leg in the hedging engine.

Conventions: `tau` is time to expiry in years; at tau <= 0 the price is the
intrinsic value, delta is the payoff indicator, gamma is 0. The normal CDF is
scipy.special.ndtr (plain ufunc — avoids the scipy 1.18 array-API dispatch
that breaks loc/scale forwarding elsewhere).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _d1_d2(spot, strike, tau_safe, sigma, r):
    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * sqrt_tau)
    return d1, d1 - sigma * sqrt_tau


def bs_price(spot, strike, tau, sigma, r=0.0, kind: str = "call") -> np.ndarray:
    """Black-Scholes price of a European call or put. tau <= 0 -> intrinsic."""
    spot, strike, tau, sigma, r = np.broadcast_arrays(spot, strike, tau, sigma, r)
    expired = tau <= 0
    tau_safe = np.where(expired, 1.0, tau)  # dummy value; result replaced below
    d1, d2 = _d1_d2(spot, strike, tau_safe, sigma, r)
    disc = np.exp(-r * tau_safe)
    call = spot * ndtr(d1) - strike * disc * ndtr(d2)
    if kind == "call":
        alive, intrinsic = call, np.maximum(spot - strike, 0.0)
    elif kind == "put":
        alive = call - spot + strike * disc  # put-call parity
        intrinsic = np.maximum(strike - spot, 0.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    result = np.where(expired, intrinsic, alive)
    return result if result.shape else float(result)


def bs_delta(spot, strike, tau, sigma, r=0.0, kind: str = "call") -> np.ndarray:
    """dPrice/dSpot. tau <= 0 -> payoff indicator (0/1 for call, -1/0 for put)."""
    spot, strike, tau, sigma, r = np.broadcast_arrays(spot, strike, tau, sigma, r)
    expired = tau <= 0
    tau_safe = np.where(expired, 1.0, tau)
    d1, _ = _d1_d2(spot, strike, tau_safe, sigma, r)
    if kind == "call":
        alive, at_expiry = ndtr(d1), (spot > strike).astype(np.float64)
    elif kind == "put":
        alive, at_expiry = ndtr(d1) - 1.0, -(spot < strike).astype(np.float64)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    result = np.where(expired, at_expiry, alive)
    return result if result.shape else float(result)


def bs_gamma(spot, strike, tau, sigma, r=0.0) -> np.ndarray:
    """d2Price/dSpot2 — same for calls and puts. tau <= 0 -> 0."""
    spot, strike, tau, sigma, r = np.broadcast_arrays(spot, strike, tau, sigma, r)
    expired = tau <= 0
    tau_safe = np.where(expired, 1.0, tau)
    d1, _ = _d1_d2(spot, strike, tau_safe, sigma, r)
    alive = _norm_pdf(d1) / (spot * sigma * np.sqrt(tau_safe))
    result = np.where(expired, 0.0, alive)
    return result if result.shape else float(result)
