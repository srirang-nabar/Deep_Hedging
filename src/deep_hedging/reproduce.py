"""Reproduction entry point: `uv run python -m deep_hedging.reproduce --tier N`.

Tier 1 (minutes): verify every artifact hash in the manifest, then execute
all committed notebooks top-to-bottom — each one loads committed artifacts
and asserts every registered claim, so a green run IS the verification.

Tier 2 (tens of minutes): recompute the published results from committed
inputs — rebuild the baseline table from the frozen TEST paths and
calibration, re-evaluate every committed policy on TEST, recompute the band
metric — and compare against the frozen artifacts to tight tolerances
(1e-9 table, 1e-6 weights-derived metrics). Exact reproduction: any drift
fails loudly.

Tier 3 (hours): retrain all policies from scratch with the documented seeds
into a scratch directory, evaluate on TEST, and check that across-seed mean
CVaR95 and turnover land inside the tolerance bands published in
results/tolerances.json. Statistical reproduction — bit-exact retraining is
impossible across hardware, and the bands (mean ± 4 across-seed std) are the
contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("0*.ipynb"))


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.0f}s"


def tier1() -> None:
    from deep_hedging import manifest

    start = time.time()
    problems = manifest.verify()
    assert not problems, f"manifest verification failed: {problems}"
    print(f"[tier1] manifest OK ({len(manifest.read_manifest())} artifacts)")
    for nb in NOTEBOOKS:
        t = time.time()
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", str(nb)],
            check=True, capture_output=True,
        )
        print(f"[tier1] {nb.name} executed clean ({_elapsed(t)})")
    print(f"[tier1] PASS in {_elapsed(start)} — all notebook assertions green")


def tier2() -> None:
    from deep_hedging.analysis import band_width, learned_no_trade_region, ww_no_trade_region
    from deep_hedging.baselines import (
        bs_delta_strategy, load_calibration, make_leland_strategy, make_whalley_wilmott_strategy,
    )
    from deep_hedging.evaluate import (
        CANONICAL, CANONICAL_GBM, build_baseline_table, cvar, generate_path_set,
        path_set_fingerprint, simulate_hedge,
    )
    from deep_hedging.policy import PolicyStrategy, load_policy
    from deep_hedging.train import COST_LEVELS_BPS, TRAIN_SEEDS, weight_path

    start = time.time()
    calib = load_calibration(RESULTS / "baseline_calibration.json")
    frozen = json.loads((RESULTS / "baseline_table.json").read_text())
    test_paths = generate_path_set("TEST", final=True)
    assert path_set_fingerprint(test_paths) == frozen["config"]["path_set_sha256"]

    factories = {
        "bs_delta": lambda bps: bs_delta_strategy,
        "leland": lambda bps: make_leland_strategy(calib["per_cost_level"][str(bps)]["leland"]["adjustment_scale"]),
        "whalley_wilmott": lambda bps: make_whalley_wilmott_strategy(calib["per_cost_level"][str(bps)]["whalley_wilmott"]["risk_aversion"]),
    }
    rebuilt = build_baseline_table(test_paths, strategy_factories=factories,
                                   meta={"path_set": "TEST", "calibration": "results/baseline_calibration.json"})
    for got, want in zip(rebuilt["rows"], frozen["rows"]):
        for key in ("mean_pnl", "std_pnl", "cvar95", "cvar99", "turnover"):
            assert abs(got[key] - want[key]) < 1e-9, (key, got["strategy"], got["cost_bps"])
    print(f"[tier2] baseline table reproduces exactly ({_elapsed(start)})")

    lp = json.loads((RESULTS / "learned_policy_results.json").read_text())
    common = dict(strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma)
    for cost in COST_LEVELS_BPS:
        for row in lp["per_cost"][str(cost)]["per_seed"]:
            policy, _ = load_policy(weight_path(cost, row["seed"]))
            res = simulate_hedge(test_paths, strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma),
                                 cost_rate=cost / 10_000.0, **common)
            assert abs(cvar(-res.pnl, 0.95) - row["cvar95"]) < 1e-6
            assert abs(res.turnover.mean() - row["turnover"]) < 1e-6
        print(f"[tier2] learned policy {cost:g} bps: all seeds reproduce to 1e-6")

    band = json.loads((RESULTS / "band_analysis.json").read_text())
    for cost, entry in band["per_cost"].items():
        ww = ww_no_trade_region(float(cost), entry["ww_risk_aversion"])
        for i, seed in enumerate(TRAIN_SEEDS):
            policy, _ = load_policy(weight_path(float(cost), seed))
            region = learned_no_trade_region(policy)
            iou = np.logical_and(region, ww).sum() / np.logical_or(region, ww).sum()
            assert abs(iou - entry["iou_per_seed"][i]) < 1e-9
            assert abs(band_width(region) - entry["width_per_seed"][i]) < 1e-9
    print(f"[tier2] band metrics reproduce exactly")
    print(f"[tier2] PASS in {_elapsed(start)}")


def tier3() -> None:
    from deep_hedging.evaluate import CANONICAL, CANONICAL_GBM, cvar, generate_path_set, simulate_hedge
    from deep_hedging.policy import PolicyStrategy, load_policy
    from deep_hedging.train import COST_LEVELS_BPS, TRAIN_SEEDS, train_policy, weight_path

    start = time.time()
    tolerances = json.loads((RESULTS / "tolerances.json").read_text())
    test_paths = generate_path_set("TEST", final=True)
    common = dict(strike=CANONICAL["strike"], horizon=CANONICAL["horizon"], sigma=CANONICAL_GBM.sigma)
    with tempfile.TemporaryDirectory() as scratch:
        out_dir = Path(scratch)
        for cost in COST_LEVELS_BPS:
            cvars, turnovers = [], []
            for seed in TRAIN_SEEDS:
                train_policy(cost, seed, out_dir=out_dir, verbose=False)
                policy, _ = load_policy(weight_path(cost, seed, out_dir))
                res = simulate_hedge(test_paths, strategy=PolicyStrategy(policy, sigma=CANONICAL_GBM.sigma),
                                     cost_rate=cost / 10_000.0, **common)
                cvars.append(cvar(-res.pnl, 0.95))
                turnovers.append(float(res.turnover.mean()))
            for metric, values in (("cvar95", cvars), ("turnover", turnovers)):
                lo, hi = tolerances[str(cost)][metric]["band"]
                mean = float(np.mean(values))
                assert lo <= mean <= hi, (
                    f"{metric} at {cost} bps: retrained mean {mean:.4f} outside published band [{lo:.4f}, {hi:.4f}]"
                )
                print(f"[tier3] {cost:g} bps {metric}: retrained mean {mean:.4f} inside band [{lo:.4f}, {hi:.4f}]")
    print(f"[tier3] PASS in {_elapsed(start)} — statistical reproduction within published bands")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args(argv)
    {1: tier1, 2: tier2, 3: tier3}[args.tier]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
