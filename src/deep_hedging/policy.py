"""Learned hedging policy.

Responsibility: the feedforward policy network mapping state
(log-moneyness, time-to-expiry fraction, current holding, running vol
estimate) to a position in [0, 1], plus save/load helpers (state_dict + JSON
sidecar with architecture, config, seed, torch version, and the fingerprints
of the path sets used in training) so committed weights reload and evaluate
on CPU, and the numpy Strategy wrapper that lets the trained policy be scored
by the SAME hedging engine as the classical baselines.

Everything runs in float64: the network is tiny, and float64 makes the
published-metrics reproduction guarantee (1e-6) trivially robust to BLAS and
thread-count differences across volunteer machines.

The running vol feature is a RiskMetrics-style EWMA of squared log-returns
(lambda = 0.94), annualized, initialized at the model sigma. On GBM it hovers
near sigma; its purpose is Stage 5, where realized vol on Heston or
bootstrapped-NIFTY paths actually moves. It must be computed IDENTICALLY in
the torch training simulation (train.py) and the numpy wrapper here — a test
pins that equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from deep_hedging.evaluate import HedgeState

EWMA_LAMBDA = 0.94
N_FEATURES = 4


def build_features(log_moneyness, time_frac, holding, vol_ratio):
    """Stack and scale the four state features. Works on numpy or torch
    inputs; scaling keeps every feature O(1) for training health:
    log-moneyness has std ~0.1-0.2, so x5; vol_ratio is (vol/sigma - 1), x5."""
    lib = torch if isinstance(log_moneyness, torch.Tensor) else np
    return lib.stack(
        [log_moneyness * 5.0, time_frac, holding, vol_ratio * 5.0],
        **({"dim": 1} if lib is torch else {"axis": 1}),
    )


class HedgePolicy(torch.nn.Module):
    """MLP: 4 features -> hidden layers (SiLU) -> sigmoid -> position in
    [0, 1]. Small on purpose: ~1.3k parameters at the default (32, 32)."""

    def __init__(self, hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        self.hidden = tuple(hidden)
        layers: list[torch.nn.Module] = []
        widths = [N_FEATURES, *hidden]
        for w_in, w_out in zip(widths[:-1], widths[1:]):
            layers += [torch.nn.Linear(w_in, w_out), torch.nn.SiLU()]
        layers.append(torch.nn.Linear(widths[-1], 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


def save_policy(policy: HedgePolicy, path: Path, sidecar: dict) -> None:
    """state_dict at <path>, JSON sidecar at <path>.json. The sidecar must
    contain everything needed to rebuild and audit the policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)
    sidecar = {"hidden": list(policy.hidden), "torch_version": torch.__version__} | sidecar
    Path(f"{path}.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def load_policy(path: Path) -> tuple[HedgePolicy, dict]:
    """Rebuild from the sidecar, load weights on CPU, return (policy, sidecar)."""
    sidecar = json.loads(Path(f"{path}.json").read_text())
    policy = HedgePolicy(hidden=tuple(sidecar["hidden"])).double()
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.eval()
    return policy, sidecar


class PolicyStrategy:
    """Adapter: HedgeState -> position, so simulate_hedge can score the
    learned policy exactly like a classical baseline.

    Stateful: it carries the EWMA vol estimate between steps (the engine
    calls steps in order) and resets itself when it sees step 0, so one
    instance can be reused across simulate_hedge calls but NOT shared across
    concurrently running engines.
    """

    def __init__(self, policy: HedgePolicy, *, sigma: float, use_inventory: bool = True):
        self.policy = policy
        self.sigma = sigma
        self.use_inventory = use_inventory  # False: ablation — hide holdings
        self._prev_spot: np.ndarray | None = None
        self._ewma_var: np.ndarray | None = None

    def __call__(self, state: HedgeState) -> np.ndarray:
        if state.step == 0:
            self._ewma_var = np.full_like(state.spot, self.sigma**2)
        else:
            log_ret = np.log(state.spot / self._prev_spot)
            self._ewma_var = (
                EWMA_LAMBDA * self._ewma_var
                + (1.0 - EWMA_LAMBDA) * log_ret**2 / state.dt
            )
        self._prev_spot = state.spot

        horizon = state.time_to_expiry + state.step * state.dt
        features = build_features(
            np.log(state.spot / state.strike),
            np.full_like(state.spot, state.time_to_expiry / horizon),
            state.holding if self.use_inventory else np.zeros_like(state.holding),
            np.sqrt(self._ewma_var) / self.sigma - 1.0,
        )
        with torch.no_grad():
            position = self.policy(torch.from_numpy(features))
        return position.numpy()
