"""Stage 5 analysis: band recovery, ablation sign-off, misspecification.

Responsibility: the quantified H2 band metric (no-trade region IoU vs the
calibrated Whalley-Wilmott band, and band-width monotonicity across cost
levels — thresholds pre-registered in HYPOTHESES.md on 2026-07-15), the
stateless-ablation TEST evaluation, and the H3 misspecification runs (Heston
paths and NIFTY stationary-block-bootstrap paths, hedged with the GBM-trained
policies and GBM-model baselines). Each `run_*` function is a sign-off
context: it may generate TEST-like path sets, and it writes a frozen JSON
artifact under results/ that notebook 05 and the gate tests re-verify.

CLI:
    uv run python -m deep_hedging.analysis band
    uv run python -m deep_hedging.analysis ablation
    uv run python -m deep_hedging.analysis misspec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deep_hedging import manifest
from deep_hedging.baselines import load_calibration, make_whalley_wilmott_strategy
from deep_hedging.evaluate import (
    CANONICAL,
    CANONICAL_GBM,
    cvar,
    generate_path_set,
    path_set_fingerprint,
    simulate_hedge,
)
from deep_hedging.policy import PolicyStrategy, build_features, load_policy
from deep_hedging.pricing import bs_delta, bs_gamma
from deep_hedging.simulate import HestonParams, simulate_heston
from deep_hedging.train import LAMBDA, TRAIN_SEEDS, weight_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
DATA = PROJECT_ROOT / "data"

# --- H2 band metric, exactly as pre-registered in HYPOTHESES.md ---
BAND_COST_LEVELS = (5.0, 20.0, 50.0)
BAND_SPOTS = np.linspace(85.0, 115.0, 61)
BAND_HOLDINGS = np.linspace(0.0, 1.0, 41)
BAND_TAU = 0.125  # mid-life
BAND_EPS = 0.01  # shares
IOU_THRESHOLD = 0.5

# --- H3 misspecification, exactly as pre-registered ---
H3_COST_BPS = 20.0
HESTON_SEED = 404
NIFTY_BOOT_SEED = 505
NIFTY_BLOCK_LENGTHS = (5, 10, 20)  # 10 is primary
N_MISSPEC_PATHS = 100_000
GBM_TEST_DIFF = -0.2230  # frozen H1 paired diff on GBM TEST (CLAIMS.md)
BOOT_SEED = 424
N_BOOT = 500


# ------------------------------------------------------------- H2: band ---


def learned_no_trade_region(policy, *, sigma: float = CANONICAL_GBM.sigma) -> np.ndarray:
    """Boolean grid (n_spots, n_holdings): |policy(S, h) - h| < eps at the
    pre-registered mid-life slice with the vol feature at its resting value."""
    spots, holdings = np.meshgrid(BAND_SPOTS, BAND_HOLDINGS, indexing="ij")
    features = build_features(
        np.log(spots / CANONICAL["strike"]).ravel(),
        np.full(spots.size, BAND_TAU / CANONICAL["horizon"]),
        holdings.ravel(),
        np.zeros(spots.size),
    )
    with torch.no_grad():
        target = policy(torch.from_numpy(features)).numpy().reshape(spots.shape)
    return np.abs(target - holdings) < BAND_EPS


def ww_no_trade_region(cost_bps: float, risk_aversion: float) -> np.ndarray:
    """The calibrated WW band on the same grid: |h - delta(S)| <= H(S)."""
    cost_rate = cost_bps / 10_000.0
    delta = bs_delta(BAND_SPOTS, CANONICAL["strike"], BAND_TAU, CANONICAL_GBM.sigma)
    gamma = bs_gamma(BAND_SPOTS, CANONICAL["strike"], BAND_TAU, CANONICAL_GBM.sigma)
    half_width = (1.5 * cost_rate * BAND_SPOTS * gamma**2 / risk_aversion) ** (1.0 / 3.0)
    return np.abs(BAND_HOLDINGS[None, :] - delta[:, None]) <= half_width[:, None]


def band_width(region: np.ndarray) -> float:
    """Mean (over the spot grid) h-measure of the no-trade region."""
    dh = BAND_HOLDINGS[1] - BAND_HOLDINGS[0]
    return float(region.sum(axis=1).mean() * dh)


def run_band_analysis() -> dict:
    calib = load_calibration(RESULTS / "baseline_calibration.json")
    per_cost: dict = {}
    widths_by_seed = {seed: [] for seed in TRAIN_SEEDS}
    for cost_bps in BAND_COST_LEVELS:
        gamma = calib["per_cost_level"][str(cost_bps)]["whalley_wilmott"]["risk_aversion"]
        ww_region = ww_no_trade_region(cost_bps, gamma)
        ious, widths = [], []
        for seed in TRAIN_SEEDS:
            policy, _ = load_policy(weight_path(cost_bps, seed))
            region = learned_no_trade_region(policy)
            inter = np.logical_and(region, ww_region).sum()
            union = np.logical_or(region, ww_region).sum()
            ious.append(float(inter / union))
            w = band_width(region)
            widths.append(w)
            widths_by_seed[seed].append(w)
        per_cost[str(cost_bps)] = {
            "ww_risk_aversion": gamma,
            "ww_width": band_width(ww_region),
            "iou_per_seed": ious,
            "iou_mean": float(np.mean(ious)),
            "width_per_seed": widths,
            "width_mean": float(np.mean(widths)),
        }

    mean_widths = [per_cost[str(c)]["width_mean"] for c in BAND_COST_LEVELS]
    means_monotone = bool(all(a < b for a, b in zip(mean_widths, mean_widths[1:])))
    seeds_monotone = sum(
        all(a < b for a, b in zip(ws, ws[1:])) for ws in widths_by_seed.values()
    )
    iou_pass = bool(all(per_cost[str(c)]["iou_mean"] >= IOU_THRESHOLD for c in BAND_COST_LEVELS))
    monotone_pass = bool(means_monotone and seeds_monotone >= 4)

    result = {
        "metric": {
            "grid": {"spots": [85.0, 115.0, 61], "holdings": [0.0, 1.0, 41],
                     "tau": BAND_TAU, "eps": BAND_EPS},
            "iou_threshold": IOU_THRESHOLD,
            "monotonicity_rule": "across-seed mean widths strictly increasing AND >=4/5 seeds strictly increasing",
        },
        "per_cost": per_cost,
        "means_monotone": means_monotone,
        "seeds_monotone_count": int(seeds_monotone),
        "iou_pass": iou_pass,
        "monotone_pass": monotone_pass,
        "h2_verdict": "supported" if (iou_pass and monotone_pass) else "not supported",
    }
    (RESULTS / "band_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest.add([RESULTS / "band_analysis.json"])
    return result


# -------------------------------------------------------- ablation (TEST) ---


def run_ablation_signoff() -> dict:
    """Evaluate the stateless (no-inventory) policies on TEST at 20 bps and
    compare with the full policies' frozen results."""
    test_paths = generate_path_set("TEST", final=True)
    common = dict(strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma)
    per_seed = []
    for seed in TRAIN_SEEDS:
        policy, sidecar = load_policy(weight_path(20.0, seed, prefix="ablation"))
        assert sidecar["mask_inventory"] is True
        res = simulate_hedge(
            test_paths,
            strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma, use_inventory=False),
            cost_rate=20.0 / 10_000.0, **common,
        )
        loss = -res.pnl
        per_seed.append({
            "seed": seed,
            "mean_pnl": float(res.pnl.mean()),
            "cvar95": cvar(loss, 0.95),
            "turnover": float(res.turnover.mean()),
        })
    full = json.loads((RESULTS / "learned_policy_results.json").read_text())["per_cost"]["20.0"]
    result = {
        "cost_bps": 20.0,
        "path_set_sha256": path_set_fingerprint(test_paths),
        "per_seed": per_seed,
        "across_seeds": {
            key: {"mean": float(np.mean([r[key] for r in per_seed])),
                  "std": float(np.std([r[key] for r in per_seed], ddof=1))}
            for key in ("mean_pnl", "cvar95", "turnover")
        },
        "full_policy_across_seeds": full["across_seeds"],
    }
    (RESULTS / "ablation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest.add([RESULTS / "ablation.json"])
    return result


# ------------------------------------------------- H3: misspecification ---


def load_nifty_log_returns() -> np.ndarray:
    closes = []
    for line in (DATA / "nifty_daily.csv").read_text().splitlines()[1:]:
        closes.append(float(line.split(",")[1]))
    return np.diff(np.log(np.array(closes)))


def bootstrap_nifty_paths(
    *, block_length: int, n_paths: int, n_steps: int, seed: int, s0: float = 100.0
) -> np.ndarray:
    """Stationary block bootstrap (Politis-Romano): stitch blocks of real
    NIFTY daily log-returns with geometric(1/L) lengths and random starts
    (wrap-around), preserving short-range dependence like vol clustering."""
    returns = load_nifty_log_returns()
    n_obs = returns.size
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, n_steps))
    starts = rng.integers(0, n_obs, size=(n_paths, n_steps))  # upper bound on block count
    lengths = rng.geometric(1.0 / block_length, size=(n_paths, n_steps))
    # per-path loop kept explicit for auditability (runs in ~seconds)
    for i in range(n_paths):
        pos = 0
        b = 0
        while pos < n_steps:
            start, length = starts[i, b], min(lengths[i, b], n_steps - pos)
            idx = (start + np.arange(length)) % n_obs
            out[i, pos : pos + length] = returns[idx]
            pos += length
            b += 1
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(out, axis=1))
    return paths


def _paired_diff(policy_losses: list[np.ndarray], ww_losses: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    n = ww_losses.size
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[b] = np.mean([cvar(losses[idx]) for losses in policy_losses]) - cvar(ww_losses[idx])
    ci = [float(q) for q in np.quantile(diffs, [0.025, 0.975])]
    return {"cvar_diff_mean": float(diffs.mean()), "cvar_diff_ci95": ci}


def _evaluate_set(paths: np.ndarray, gamma: float) -> dict:
    common = dict(strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma)
    cost_rate = H3_COST_BPS / 10_000.0
    ww = simulate_hedge(paths, strategy=make_whalley_wilmott_strategy(gamma), cost_rate=cost_rate, **common)
    policy_losses, per_seed = [], []
    for seed in TRAIN_SEEDS:
        policy, _ = load_policy(weight_path(H3_COST_BPS, seed))
        res = simulate_hedge(
            paths, strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma),
            cost_rate=cost_rate, **common,
        )
        policy_losses.append(-res.pnl)
        per_seed.append({"seed": seed, "cvar95": cvar(-res.pnl, 0.95), "turnover": float(res.turnover.mean())})
    stats = _paired_diff(policy_losses, -ww.pnl)
    return stats | {
        "policy_cvar95_mean": float(np.mean([r["cvar95"] for r in per_seed])),
        "policy_cvar95_std": float(np.std([r["cvar95"] for r in per_seed], ddof=1)),
        "ww_cvar95": cvar(-ww.pnl, 0.95),
        "per_seed": per_seed,
        "path_fingerprint": path_set_fingerprint(paths),
    }


def run_misspecification() -> dict:
    calib = load_calibration(RESULTS / "baseline_calibration.json")
    gamma = calib["per_cost_level"][str(H3_COST_BPS)]["whalley_wilmott"]["risk_aversion"]
    n_steps = CANONICAL["n_steps"]

    heston = simulate_heston(
        HestonParams(), n_paths=N_MISSPEC_PATHS, n_steps=n_steps,
        horizon=CANONICAL["horizon"], seed=HESTON_SEED,
    )
    sets = {"heston": _evaluate_set(heston.spot, gamma)}
    for block in NIFTY_BLOCK_LENGTHS:
        paths = bootstrap_nifty_paths(
            block_length=block, n_paths=N_MISSPEC_PATHS, n_steps=n_steps, seed=NIFTY_BOOT_SEED
        )
        sets[f"nifty_block{block}"] = _evaluate_set(paths, gamma)

    primary = [sets["heston"], sets["nifty_block10"]]
    not_inverted = all(s["cvar_diff_ci95"][1] < 0.0 for s in primary)
    shrunk = all(s["cvar_diff_mean"] > GBM_TEST_DIFF for s in primary)
    result = {
        "config": {
            "cost_bps": H3_COST_BPS, "n_paths": N_MISSPEC_PATHS,
            "heston_seed": HESTON_SEED, "nifty_boot_seed": NIFTY_BOOT_SEED,
            "block_lengths": list(NIFTY_BLOCK_LENGTHS), "primary_block": 10,
            "gbm_test_diff": GBM_TEST_DIFF, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "nifty_sha256": manifest.read_manifest().get("data/nifty_daily.csv"),
        },
        "sets": sets,
        "not_inverted": bool(not_inverted),
        "shrunk": bool(shrunk),
        "h3_verdict": "supported" if (not_inverted and shrunk) else "not supported",
    }
    (RESULTS / "misspecification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest.add([RESULTS / "misspecification.json"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["band", "ablation", "misspec"])
    args = parser.parse_args(argv)
    result = {"band": run_band_analysis, "ablation": run_ablation_signoff, "misspec": run_misspecification}[args.what]()
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict) or k == "across_seeds"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
