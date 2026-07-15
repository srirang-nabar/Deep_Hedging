"""Policy training and sign-off.

Responsibility: train the policy by differentiating directly through
simulated P&L on TRAIN paths — the simulator is a chain of elementary torch
ops, so the mean-CVaR objective is differentiable end-to-end and no
policy-gradient machinery is needed. The torch simulation mirrors the numpy
engine (evaluate.simulate_hedge) exactly at r=0; a test pins the equivalence,
so training-time P&L and published evaluation P&L are the same quantity.

Protocol (plan.md Stage 4): objective = mean loss + lambda * CVaR95 loss with
lambda = 1 — the SAME objective the classical baselines were calibrated on.
>= 5 seeds per cost level; early stopping on the VAL objective only; TEST is
touched exclusively by `sign_off`, which evaluates committed weights via the
numpy engine, runs the paired H1 comparison against calibrated
Whalley-Wilmott, and writes results/learned_policy_results.json +
results/tolerances.json (the Tier 3 retraining contract).

CLI:
    uv run python -m deep_hedging.train train --cost-bps 20 --seed 0 [--smoke]
    uv run python -m deep_hedging.train train-all [--smoke]
    uv run python -m deep_hedging.train sign-off [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
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
from deep_hedging.policy import EWMA_LAMBDA, HedgePolicy, PolicyStrategy, build_features, load_policy, save_policy
from deep_hedging.pricing import bs_price

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
WEIGHTS_DIR = RESULTS / "weights"

COST_LEVELS_BPS = (5.0, 20.0, 50.0)  # 0 bps omitted: delta hedging is already
# optimal there and the zero-cost engine equivalences cover that regime
TRAIN_SEEDS = (0, 1, 2, 3, 4)
LAMBDA = 1.0  # mean-CVaR tradeoff — identical to baseline calibration


def weight_path(cost_bps: float, seed: int, out_dir: Path = WEIGHTS_DIR, prefix: str = "policy") -> Path:
    return out_dir / f"{prefix}_c{cost_bps:g}_s{seed}.pt"


def cvar_torch(losses: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Differentiable CVaR: mean of the worst ceil((1-alpha) n) losses via
    topk — same tail count rule (with the same float-epsilon guard) as
    evaluate.cvar, so train-time and published CVaR agree on a full pass."""
    k = max(1, int(np.ceil((1.0 - alpha) * losses.shape[0] - 1e-9)))
    return torch.topk(losses, k).values.mean()


def objective_torch(pnl: torch.Tensor, *, lam: float = LAMBDA) -> torch.Tensor:
    loss = -pnl
    return loss.mean() + lam * cvar_torch(loss)


def simulate_pnl_torch(
    policy: HedgePolicy,
    paths: torch.Tensor,
    *,
    strike: float = CANONICAL["strike"],
    horizon: float = CANONICAL["horizon"],
    sigma: float = CANONICAL_GBM.sigma,
    cost_rate: float,
    mask_inventory: bool = False,
) -> torch.Tensor:
    """Differentiable mirror of evaluate.simulate_hedge at r=0, including the
    terminal liquidation cost and the EWMA vol feature of PolicyStrategy.
    mask_inventory=True zeroes the holding feature (stateless ablation)."""
    n_steps = paths.shape[1] - 1
    dt = horizon / n_steps
    premium = float(bs_price(float(paths[0, 0]), strike, horizon, sigma))

    holding = torch.zeros(paths.shape[0], dtype=paths.dtype)
    ewma_var = torch.full_like(holding, sigma**2)
    trading_pnl = torch.zeros_like(holding)
    costs = torch.zeros_like(holding)

    for k in range(n_steps):
        spot = paths[:, k]
        if k > 0:
            log_ret = torch.log(spot / paths[:, k - 1])
            ewma_var = EWMA_LAMBDA * ewma_var + (1.0 - EWMA_LAMBDA) * log_ret**2 / dt
        tau = horizon - k * dt
        features = build_features(
            torch.log(spot / strike),
            torch.full_like(spot, tau / horizon),
            holding * 0.0 if mask_inventory else holding,
            torch.sqrt(ewma_var) / sigma - 1.0,
        )
        target = policy(features)
        costs = costs + cost_rate * (target - holding).abs() * spot
        trading_pnl = trading_pnl + target * (paths[:, k + 1] - spot)
        holding = target

    terminal = paths[:, -1]
    payoff = torch.clamp(terminal - strike, min=0.0)
    costs = costs + cost_rate * holding.abs() * terminal
    return premium + trading_pnl - costs - payoff


def train_policy(
    cost_bps: float,
    seed: int,
    *,
    smoke: bool = False,
    out_dir: Path = WEIGHTS_DIR,
    verbose: bool = True,
    mask_inventory: bool = False,
    prefix: str = "policy",
) -> dict:
    """Train one policy; save weights + sidecar; return the sidecar dict.

    Hygiene: learns from TRAIN batches only; after every epoch the objective
    is measured on a VAL prefix, and the best-VAL weights are kept (early
    stopping with patience). TEST cannot even be generated here — the
    evaluate-module guard raises without final=True.
    """
    torch.manual_seed(seed)
    n_train, n_val = (2_000, 1_000) if smoke else (100_000, 20_000)
    max_epochs, patience = (3, 2) if smoke else (120, 12)
    batch_size = 512 if smoke else 8_192
    lr = 3e-3

    train_paths = generate_path_set("TRAIN", n_paths=n_train)
    val_paths = generate_path_set("VAL", n_paths=n_val)
    # float32 for training speed; the published float64 evaluation happens on
    # the saved weights (cast at save time), so reproducibility is unaffected
    t_train = torch.from_numpy(train_paths).float()
    t_val = torch.from_numpy(val_paths).float()
    cost_rate = cost_bps / 10_000.0

    policy = HedgePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr / 30)
    shuffle_rng = np.random.default_rng(seed)

    best_val = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    best_epoch = -1
    train_curve: list[float] = []
    val_curve: list[float] = []
    start = time.time()

    for epoch in range(max_epochs):
        order = shuffle_rng.permutation(n_train)
        epoch_losses = []
        for lo in range(0, n_train, batch_size):
            batch = t_train[order[lo : lo + batch_size]]
            loss = objective_torch(simulate_pnl_torch(policy, batch, cost_rate=cost_rate, mask_inventory=mask_inventory))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        train_curve.append(float(np.mean(epoch_losses)))
        scheduler.step()

        with torch.no_grad():
            val_j = float(objective_torch(simulate_pnl_torch(policy, t_val, cost_rate=cost_rate, mask_inventory=mask_inventory)))
        val_curve.append(val_j)
        if verbose:
            print(f"  c{cost_bps:g} s{seed} epoch {epoch:02d}: train J {train_curve[-1]:.4f}  val J {val_j:.4f}")
        if val_j < best_val - 1e-6:
            best_val, best_epoch = val_j, epoch
            best_state = copy.deepcopy(policy.state_dict())
        elif epoch - best_epoch >= patience:
            break

    policy.load_state_dict(best_state)
    policy = policy.double()
    sidecar = {
        "cost_bps": cost_bps,
        "seed": seed,
        "smoke": smoke,
        "mask_inventory": mask_inventory,
        "objective": {"kind": "mean_cvar", "alpha": 0.95, "lam": LAMBDA},
        "optimizer": {"name": "adam", "lr": lr, "schedule": "cosine", "batch_size": batch_size, "train_dtype": "float32"},
        "max_epochs": max_epochs,
        "patience": patience,
        "epochs_trained": len(val_curve),
        "best_epoch": best_epoch,
        "best_val_objective": best_val,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "wall_seconds": round(time.time() - start, 1),
        "trained_on": {
            "TRAIN": {"n_paths": n_train, "sha256": path_set_fingerprint(train_paths)},
            "VAL": {"n_paths": n_val, "sha256": path_set_fingerprint(val_paths)},
        },
        "market": {"gbm": vars(CANONICAL_GBM)} | CANONICAL,
    }
    save_policy(policy, weight_path(cost_bps, seed, out_dir, prefix), sidecar)
    return sidecar


def train_all(*, smoke: bool = False, out_dir: Path = WEIGHTS_DIR) -> None:
    for cost_bps in COST_LEVELS_BPS:
        for seed in TRAIN_SEEDS:
            sidecar = train_policy(cost_bps, seed, smoke=smoke, out_dir=out_dir)
            print(
                f"trained c{cost_bps:g} s{seed}: best val J {sidecar['best_val_objective']:.4f} "
                f"(epoch {sidecar['best_epoch']}, {sidecar['wall_seconds']}s)"
            )


def _summarize_losses(pnl: np.ndarray, turnover: np.ndarray) -> dict:
    loss = -pnl
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
        "cvar95": cvar(loss, 0.95),
        "cvar99": cvar(loss, 0.99),
        "turnover": float(turnover.mean()),
    }


def sign_off(*, n_boot: int = 500, boot_seed: int = 424, out_dir: Path = WEIGHTS_DIR) -> dict:
    """The only code path that touches TEST. Evaluates every committed policy
    on the frozen TEST set via the numpy engine, aggregates across seeds,
    runs the paired bootstrap for H1 at 20 bps vs calibrated WW, writes
    results/learned_policy_results.json and results/tolerances.json, and
    freezes weights + results into the manifest."""
    calib = load_calibration(RESULTS / "baseline_calibration.json")
    test_paths = generate_path_set("TEST", final=True)
    common = dict(
        strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma
    )

    results: dict = {
        "config": {
            "path_set": "TEST",
            "path_set_sha256": path_set_fingerprint(test_paths),
            "n_boot": n_boot,
            "boot_seed": boot_seed,
            "seeds": list(TRAIN_SEEDS),
            "lambda": LAMBDA,
        },
        "per_cost": {},
    }
    tolerances: dict = {}
    h1_losses_policy: list[np.ndarray] = []
    h1_ww: dict = {}

    for cost_bps in COST_LEVELS_BPS:
        cost_rate = cost_bps / 10_000.0
        gamma = calib["per_cost_level"][str(cost_bps)]["whalley_wilmott"]["risk_aversion"]
        ww_res = simulate_hedge(
            test_paths, strategy=make_whalley_wilmott_strategy(gamma),
            cost_rate=cost_rate, **common,
        )
        per_seed = []
        for seed in TRAIN_SEEDS:
            policy, sidecar = load_policy(weight_path(cost_bps, seed, out_dir))
            assert set(sidecar["trained_on"]) <= {"TRAIN", "VAL"}, "leak: non-dev path set in training"
            res = simulate_hedge(
                test_paths, strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma),
                cost_rate=cost_rate, **common,
            )
            per_seed.append({"seed": seed} | _summarize_losses(res.pnl, res.turnover))
            if cost_bps == 20.0:
                h1_losses_policy.append(-res.pnl)

        across = {
            key: {
                "mean": float(np.mean([r[key] for r in per_seed])),
                "std": float(np.std([r[key] for r in per_seed], ddof=1)),
            }
            for key in ("mean_pnl", "cvar95", "cvar99", "turnover")
        }
        entry = {
            "per_seed": per_seed,
            "across_seeds": across,
            "ww_baseline": _summarize_losses(ww_res.pnl, ww_res.turnover) | {"risk_aversion": gamma},
        }
        results["per_cost"][str(cost_bps)] = entry
        tolerances[str(cost_bps)] = {
            key: {
                "mean": across[key]["mean"],
                "std": across[key]["std"],
                "band": [
                    across[key]["mean"] - 4 * across[key]["std"],
                    across[key]["mean"] + 4 * across[key]["std"],
                ],
            }
            for key in ("cvar95", "turnover")
        }
        if cost_bps == 20.0:
            h1_ww = {"losses": -ww_res.pnl, "turnover": float(ww_res.turnover.mean())}

    # H1: paired bootstrap at 20 bps — resample the SAME path indices for the
    # policy (averaged across seeds) and WW, so path-level noise cancels.
    rng = np.random.default_rng(boot_seed)
    n = h1_ww["losses"].size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        policy_cvar = np.mean([cvar(losses[idx]) for losses in h1_losses_policy])
        diffs[b] = policy_cvar - cvar(h1_ww["losses"][idx])
    ci = [float(q) for q in np.quantile(diffs, [0.025, 0.975])]
    entry20 = results["per_cost"]["20.0"]
    turnover_ok = entry20["across_seeds"]["turnover"]["mean"] <= h1_ww["turnover"]
    supported = ci[1] < 0.0 and turnover_ok
    results["h1"] = {
        "statement": "at 20 bps the learned policy reduces CVaR95 vs calibrated WW at equal or lower turnover",
        "cvar_diff_mean": float(diffs.mean()),
        "cvar_diff_ci95": ci,
        "policy_turnover_mean": entry20["across_seeds"]["turnover"]["mean"],
        "ww_turnover": h1_ww["turnover"],
        "turnover_condition_met": bool(turnover_ok),
        "verdict": "supported" if supported else "not supported",
    }

    (RESULTS / "learned_policy_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULTS / "tolerances.json").write_text(json.dumps(tolerances, indent=2, sort_keys=True) + "\n")

    artifacts = [RESULTS / "learned_policy_results.json", RESULTS / "tolerances.json"]
    for cost_bps in COST_LEVELS_BPS:
        for seed in TRAIN_SEEDS:
            p = weight_path(cost_bps, seed, out_dir)
            artifacts += [p, Path(f"{p}.json")]
    manifest.add(artifacts, allow_change=False)
    print(json.dumps(results["h1"], indent=2))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--cost-bps", type=float, required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--smoke", action="store_true")
    p_all = sub.add_parser("train-all")
    p_all.add_argument("--smoke", action="store_true")
    sub.add_parser("sign-off")
    args = parser.parse_args(argv)

    if args.command == "train":
        train_policy(args.cost_bps, args.seed, smoke=args.smoke)
    elif args.command == "train-all":
        train_all(smoke=args.smoke)
    else:
        sign_off()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
