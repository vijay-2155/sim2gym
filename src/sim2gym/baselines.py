"""Non-LLM baselines. They drive the same Sim2GymEnv (same protocol text, same grading path).

Baselines are labeled ``baseline: true`` in results and must never be presented as model inference.
- oracle:        answers the stored exhaustive optimum (upper bound; it reads the task's optimum).
- random:        one seeded random feasible design, no experiments (lower bound).
- max-square-ws: largest square array within the PE budget, ws dataflow, balanced SRAM split.
- 6-probe:       up to 6 seeded random distinct feasible designs simulated through the env,
                 then answers the one with the fewest observed cycles.
"""
from __future__ import annotations

import hashlib
import json
import random

from sim2gym.packs import accel

BASELINES = ["oracle", "random", "max-square-ws", "6-probe"]


def _rng(task_id: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _say(kind: str, design: dict, note: str) -> str:
    return f"{note}\n{json.dumps({kind: design})}"


def _default_split() -> str:
    return "balanced" if "balanced" in accel.SRAM_SPLITS else accel.SRAM_SPLITS[0]


class Baseline:
    is_baseline = True
    name = "baseline"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def settings(self) -> dict:
        return {"provider": "baseline", "baseline": self.name, "seed": self.seed}

    def start(self, task: dict) -> None:
        self.task = task

    def act(self, obs: str, info: dict | None) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class Oracle(Baseline):
    name = "oracle"

    def act(self, obs, info):
        return _say("answer", dict(self.task["optimum"]["inputs"]), "Oracle baseline: stored exhaustive optimum.")


class RandomDesign(Baseline):
    name = "random"

    def act(self, obs, info):
        designs = accel.feasible_designs(int(self.task["pe_budget"]))
        return _say("answer", _rng(self.task["id"], self.seed).choice(designs), "Random baseline.")


def max_square_ws(pe_budget: int) -> dict:
    side = max((d for d in accel.ARRAY_DIMS if d * d <= pe_budget), default=min(accel.ARRAY_DIMS))
    return {"array_h": side, "array_w": side, "dataflow": "ws", "sram_split": _default_split()}


class MaxSquareWS(Baseline):
    name = "max-square-ws"

    def act(self, obs, info):
        return _say("answer", max_square_ws(int(self.task["pe_budget"])), "Max-square ws baseline.")


class Probe(Baseline):
    name = "6-probe"
    probes = 6

    def start(self, task):
        super().start(task)
        designs = list(accel.feasible_designs(int(task["pe_budget"])))
        _rng(task["id"], self.seed).shuffle(designs)
        n = min(self.probes, int(task.get("budget", self.probes)), len(designs))
        self.queue = designs[:n]

    def act(self, obs, info):
        if self.queue:
            return _say("simulate", self.queue.pop(0), "6-probe baseline: random probe.")
        log = [e for e in (info or {}).get("sim_log", []) if isinstance(e.get("cycles"), (int, float))]
        if log:
            best = min(log, key=lambda e: e["cycles"])["design"]
        else:  # no successful probe: fall back to a seeded random feasible design
            best = _rng(self.task["id"], self.seed).choice(accel.feasible_designs(int(self.task["pe_budget"])))
        return _say("answer", best, "6-probe baseline: best probe.")


_REGISTRY = {c.name: c for c in (Oracle, RandomDesign, MaxSquareWS, Probe)}


def get_baseline(name: str, seed: int = 0) -> Baseline:
    if name not in _REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
    return _REGISTRY[name](seed=seed)
