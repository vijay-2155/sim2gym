"""Sim2GymEnv rules: budget, invalid simulate, retries, max turns, terminal, malformed/NaN JSON.

Uses the real protocol / grading / tasks / validate_design with a fast deterministic fake simulator.
"""
import json

import pytest

from sim2gym import protocol, tasks
from sim2gym.env import Sim2GymEnv
from sim2gym.packs import accel

WL = "bert_tiny_attention_proj_s8"


def fake_simulate(layer, array_h, array_w, dataflow, sram_split):
    cycles = 1000 + 7 * abs(array_h - 2 * array_w) + {"os": 0, "ws": 50, "is": 100}[dataflow] + 20000 // (array_h * array_w)
    return {"cycles": cycles, "stall_cycles": 0, "utilization_pct": 50.0}


@pytest.fixture
def task(monkeypatch):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    return tasks.build_tasks(seed=0, budget=6, workloads=[WL], pe_budgets=[256])[0]


def J(kind, **d):
    base = {"array_h": 8, "array_w": 8, "dataflow": "os", "sram_split": "balanced"}
    base.update(d)
    return f"thinking...\n{json.dumps({kind: base})}"


def test_reset_and_simulate(task):
    env = Sim2GymEnv([task])
    prompt = env.reset(0)
    assert prompt == task["prompt"] and env.budget == 6 and env.max_turns == 8
    obs, r, done, info = env.step(J("simulate"))
    assert not done and r == 0.0 and json.loads(obs)["cycles"] == fake_simulate(None, 8, 8, "os", "")["cycles"]
    assert info["sim_calls_used"] == 1 and info["turns"] == 1
    assert info["sim_log"][0]["design"]["array_h"] == 8 and "cycles" in info["sim_log"][0]
    assert info["event"]["type"] == "simulate"


def test_oracle_answer_scores_one_and_terminal_rejects(task):
    env = Sim2GymEnv([task])
    env.reset(0)
    obs, r, done, info = env.step(json.dumps({"answer": task["optimum"]["inputs"]}))
    assert done and r == 1.0 and info["grade"]["solved"] and info["grade"]["feasible"]
    with pytest.raises(RuntimeError):
        env.step(J("answer"))


def test_infeasible_answer_zero(task):
    env = Sim2GymEnv([task])
    env.reset(0)
    _, r, done, info = env.step(J("answer", array_h=64, array_w=64))
    assert done and r == 0.0 and info["grade"]["feasible"] is False


def test_invalid_simulate_uses_turn_not_budget_and_is_bounded(task):
    env = Sim2GymEnv([task])
    env.reset(0)
    for i in range(env.max_turns):
        obs, r, done, info = env.step(J("simulate", array_h=64, array_w=64))  # over PE budget
        assert "error" in json.loads(obs.splitlines()[0])
        assert info["sim_calls_used"] == 0
    assert done and r == 0.0 and info["turns"] == env.max_turns == 8
    assert "max turns" in info["error"]


def test_budget_exhausted_is_error_turn(task):
    env = Sim2GymEnv([task], budget=2)
    prompt = env.reset(0)
    assert "at most 2 simulator calls" in prompt  # prompt re-rendered with the enforced budget
    env.step(J("simulate"))
    env.step(J("simulate", array_w=16))
    obs, r, done, info = env.step(J("simulate", array_w=32))
    assert not done and "exhausted" in json.loads(obs)["error"] and info["sim_calls_used"] == 2
    assert info["event"]["type"] == "budget_exhausted"
    _, r, done, info = env.step(J("answer"))  # turn 4 == max_turns: answer still graded
    assert done and info["grade"]["feasible"] and info["turns"] == 4


def test_max_turns_without_answer(task):
    env = Sim2GymEnv([task], budget=1)
    env.reset(0)
    env.step(J("simulate"))
    env.step(J("simulate"))  # exhausted
    _, r, done, info = env.step(J("simulate"))
    assert done and r == 0.0 and info["turns"] == 3


def test_one_malformed_retry_then_zero(task):
    env = Sim2GymEnv([task])
    env.reset(0)
    obs, r, done, info = env.step("I think 8x8 is good")
    assert obs == protocol.RETRY_MESSAGE and not done and info["retries_used"] == 1
    env.step(J("simulate"))  # retry budget is per episode, not per streak
    obs, r, done, info = env.step("still no json")
    assert done and r == 0.0 and "malformed" in info["error"]


@pytest.mark.parametrize("bad", [
    '{"answer": {"array_h": NaN, "array_w": 8, "dataflow": "os", "sram_split": "balanced"}}',
    '{"answer": {"array_h": Infinity, "array_w": 8, "dataflow": "os", "sram_split": "balanced"}}',
    '{"simulate": {}, "answer": {}}',
    '',
])
def test_nan_and_bad_json_are_malformed(task, bad):
    env = Sim2GymEnv([task])
    env.reset(0)
    obs, r, done, info = env.step(bad)
    assert obs == protocol.RETRY_MESSAGE and info["event"]["type"] == "malformed"


def test_non_dict_and_bool_designs_rejected(task):
    env = Sim2GymEnv([task])
    env.reset(0)
    obs, _, done, info = env.step('{"simulate": [8, 8]}')
    assert "error" in json.loads(obs) and info["sim_calls_used"] == 0
    obs, _, done, info = env.step(J("simulate", array_h=True))
    assert "error" in json.loads(obs) and info["sim_calls_used"] == 0
    _, r, done, info = env.step('{"answer": "8x8"}')
    assert done and r == 0.0


def test_max_turns_override_and_bad_args(task):
    env = Sim2GymEnv([task], max_turns=2)
    env.reset(0)
    env.step(J("simulate"))
    _, r, done, info = env.step(J("simulate", array_w=16))
    assert done and r == 0.0 and info["sim_calls_used"] == 2
    with pytest.raises(ValueError):
        Sim2GymEnv([task], max_turns=0)
    with pytest.raises(IndexError):
        Sim2GymEnv([task]).reset(5)
    with pytest.raises(RuntimeError):
        Sim2GymEnv([task]).step("x")  # not reset
