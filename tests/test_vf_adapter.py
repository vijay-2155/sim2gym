"""verifiers v1 adapter: taskset loading and full episodes through Env.run with scripted replies."""
import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")

import sim2gym_vf  # noqa: E402
from sim2gym import tasks  # noqa: E402
from sim2gym.packs import accel  # noqa: E402

from test_env import WL, fake_simulate  # noqa: E402


class Trace:
    def __init__(self):
        self.rewards, self.metrics, self.info = {}, {}, {}

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value

    def record_metric(self, name, value):
        self.metrics[name] = value


class ScriptedPlayer:
    """Stands in for `agents.player`: replies from a script, records what the env sent back."""

    def __init__(self, replies, prompt):
        self.replies, self.prompt, self.sent = list(replies), prompt, []

    @contextlib.asynccontextmanager
    async def interaction(self, task):
        assert task.data.prompt == self.prompt
        trace = Trace()

        async def turn(message=None):
            if message is not None:
                self.sent.append(message)
            if not self.replies:
                return vf.Segment(messages=[], terminated=True)
            return vf.Segment(messages=[vf.AssistantMessage(content=self.replies.pop(0))])

        yield SimpleNamespace(turn=turn, trace=trace)


def _taskset(tmp_path, monkeypatch):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    path = tmp_path / "tasks.jsonl"
    tasks.save_tasks(path, tasks.build_tasks(seed=0, workloads=[WL], pe_budgets=[256, 1024]))
    return sim2gym_vf.Sim2GymTaskset(sim2gym_vf.Sim2GymConfig(id="sim2gym-vf", path=str(path)))


def _play(task, replies):
    player = ScriptedPlayer(replies, task.data.prompt)
    env = object.__new__(sim2gym_vf.Sim2GymVfEnv)  # run() reads no env config
    trace_box = {}
    original = player.interaction

    @contextlib.asynccontextmanager
    async def capture(t):
        async with original(t) as it:
            trace_box["trace"] = it.trace
            yield it

    player.interaction = capture
    asyncio.run(env.run(task, SimpleNamespace(player=player)))
    return trace_box["trace"], player.sent


def J(kind, design):
    return f"reasoning...\n{json.dumps({kind: design})}"


def test_taskset_loads_tasks_with_prompts(tmp_path, monkeypatch):
    loaded = list(_taskset(tmp_path, monkeypatch))
    assert len(loaded) == 2 and {t.data.task["pe_budget"] for t in loaded} == {256, 1024}
    for t in loaded:
        assert t.data.prompt == t.data.task["prompt"] and "optimum" not in t.data.prompt.lower()


def test_simulate_then_optimal_answer_scores_one(tmp_path, monkeypatch):
    task = list(_taskset(tmp_path, monkeypatch))[0]
    best = task.data.task["optimum"]["inputs"]
    trace, sent = _play(task, [J("simulate", best), J("answer", best)])
    assert trace.rewards["sim2gym_score"] == 1.0
    assert trace.metrics == {"solved": 1.0, "feasible": 1.0, "sim_calls_used": 1.0}
    assert json.loads(sent[0])["cycles"] == task.data.task["optimum"]["value"]
    assert trace.info["sim2gym"]["answer"] == best and len(trace.info["sim2gym"]["sim_log"]) == 1


def test_infeasible_answer_scores_zero(tmp_path, monkeypatch):
    task = [t for t in _taskset(tmp_path, monkeypatch) if t.data.task["pe_budget"] == 256][0]
    trace, _ = _play(task, [J("answer", dict(array_h=64, array_w=64, dataflow="os", sram_split="balanced"))])
    assert trace.rewards["sim2gym_score"] == 0.0 and trace.metrics["feasible"] == 0.0


def test_malformed_twice_ends_with_zero(tmp_path, monkeypatch):
    task = list(_taskset(tmp_path, monkeypatch))[0]
    trace, sent = _play(task, ["no json here", "still none"])
    assert trace.rewards["sim2gym_score"] == 0.0 and "Invalid reply" in sent[0]


def test_harness_ending_early_scores_zero(tmp_path, monkeypatch):
    task = list(_taskset(tmp_path, monkeypatch))[0]
    trace, _ = _play(task, [J("simulate", task.data.task["optimum"]["inputs"])])
    assert trace.rewards["sim2gym_score"] == 0.0 and trace.metrics["sim_calls_used"] == 1.0


def test_generated_taskset(monkeypatch):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    ts = sim2gym_vf.Sim2GymTaskset(sim2gym_vf.Sim2GymConfig(id="sim2gym-vf", generate_seed=5, generate_n=2))
    loaded = list(ts)
    assert len(loaded) == 2 * len(accel.PE_BUDGETS)
    assert all(t.data.task["layer_name"].startswith("gen5_") for t in loaded)


def test_verifiers_resolves_package_by_id():
    from verifiers.v1.utils import loaders
    module = loaders.import_taskset("sim2gym-vf")
    assert loaders._plugin_class(module, vf.Taskset, "taskset") is sim2gym_vf.Sim2GymTaskset
    assert loaders._plugin_class(module, vf.Env, "env") is sim2gym_vf.Sim2GymVfEnv
