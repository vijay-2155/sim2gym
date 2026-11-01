"""Sim2GymEnv: the multi-turn, budgeted black-box environment.

Shared by evaluation (runner), baselines, the HTTP sessions and the live demo, so every
path scores answers the same way.

Turn / budget rules (PROJECT_CONTEXT 4.3-4.4, BUILD_PLAN step 3):
- Every ``step`` call is one turn. ``max_turns = budget + 2``. Hitting max turns without an
  answer ends the episode with score 0.
- ``{"simulate": design}`` with a *valid* design runs the real simulator and uses one budget unit.
- ``{"simulate": design}`` with an *invalid* design returns an error observation. It does NOT use a
  budget unit (budget counts real simulator runs) but it does use a turn, so invalid attempts are
  bounded by ``max_turns`` and can never buy unlimited experiments.
- ``simulate`` after the budget is exhausted returns an error observation (uses a turn).
- A reply with no parseable protocol JSON gets one retry message per episode (uses a turn); the
  second malformed reply ends the episode with score 0.
- ``{"answer": design}`` is graded with ``grading.grade_design`` and ends the episode.
- A terminal episode rejects further steps (``RuntimeError``).
- ``max_turns`` may be lowered explicitly (used by the paid-model guard for 1-2 turn smoke tests).
"""
from __future__ import annotations

import copy
from typing import Any

from sim2gym import grading, protocol
from sim2gym.packs import accel

DEFAULT_BUDGET = 6


def _design_view(design: Any) -> Any:
    return copy.deepcopy(design) if isinstance(design, (dict, list)) else design


class Sim2GymEnv:
    def __init__(self, tasks: list[dict], budget: int | None = DEFAULT_BUDGET, max_turns: int | None = None):
        if not tasks:
            raise ValueError("Sim2GymEnv needs at least one task")
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 0):
            raise ValueError("budget must be a non-negative int")
        self.tasks = tasks
        self.budget_override = budget
        if max_turns is not None and (not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1):
            raise ValueError("max_turns must be a positive int")
        self.max_turns_override = max_turns
        self.task: dict | None = None
        self.done = True

    # ------------------------------------------------------------------ helpers
    def _prepare_task(self, task: dict) -> dict:
        budget = self.budget_override
        if budget is None:
            budget = int(task.get("budget", DEFAULT_BUDGET))
        if task.get("budget") == budget and task.get("prompt"):
            return task
        t = dict(task)
        t["budget"] = budget
        t["prompt"] = protocol.render_prompt(t)  # prompt must state the budget actually enforced
        return t

    def _info(self, event: dict) -> dict:
        return {
            "task_id": self.task["id"] if self.task else None,
            "event": event,
            "sim_calls_used": self.sim_calls_used,
            "budget": self.budget,
            "turns": self.turns,
            "max_turns": self.max_turns,
            "retries_used": self.retries_used,
            "done": self.done,
            "answer": _design_view(self.answer),
            "grade": copy.deepcopy(self.grade),
            "error": self.error,
            "sim_log": copy.deepcopy(self.sim_log),
        }

    def _finish(self, grade: dict, error: str | None = None) -> float:
        self.done = True
        self.grade = grade
        self.error = error if error is not None else grade.get("error")
        return float(grade.get("score", 0.0))

    @staticmethod
    def _zero_grade(error: str) -> dict:
        return {"score": 0.0, "solved": False, "feasible": False, "cycles": None, "error": error}

    # ------------------------------------------------------------------ API
    def reset(self, i: int = 0) -> str:
        if not isinstance(i, int) or not 0 <= i < len(self.tasks):
            raise IndexError(f"task index {i} out of range (0..{len(self.tasks) - 1})")
        self.task = self._prepare_task(self.tasks[i])
        self.budget = int(self.task["budget"])
        self.max_turns = self.budget + 2 if self.max_turns_override is None else self.max_turns_override
        self.sim_calls_used = 0
        self.turns = 0
        self.retries_used = 0
        self.sim_log: list[dict] = []
        self.answer: Any = None
        self.grade: dict | None = None
        self.error: str | None = None
        self.done = False
        return self.task["prompt"]

    def step(self, text: str) -> tuple[str, float, bool, dict]:
        if self.task is None or self.done:
            raise RuntimeError("episode is terminal; call reset() first")
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        self.turns += 1
        obs, reward, event = self._handle(text)
        if not self.done and self.turns >= self.max_turns:
            reward = self._finish(self._zero_grade("max turns reached without an answer"))
            event = dict(event, max_turns_reached=True)
            obs = obs + "\n" + protocol.format_observation({"error": "max turns reached; episode over, score 0"})
        return obs, reward, self.done, self._info(event)

    def _handle(self, text: str) -> tuple[str, float, dict]:
        parsed = protocol.parse_reply(text)
        if parsed is None:
            if self.retries_used == 0:
                self.retries_used = 1
                return protocol.RETRY_MESSAGE, 0.0, {"type": "malformed", "retry": True}
            reward = self._finish(self._zero_grade("malformed reply after retry"))
            return protocol.format_observation({"error": "malformed reply again; episode over, score 0"}), reward, {
                "type": "malformed", "retry": False}

        if "answer" in parsed:
            design = parsed["answer"]
            self.answer = _design_view(design)
            grade = grading.grade_design(self.task, design)
            reward = self._finish(grade)
            obs = protocol.format_observation({"final": True, "score": grade.get("score"), "feasible": grade.get("feasible"),
                                               "cycles": grade.get("cycles"), "error": grade.get("error")})
            return obs, reward, {"type": "answer", "design": _design_view(design), "grade": copy.deepcopy(grade)}

        design = parsed.get("simulate")
        left = self.budget - self.sim_calls_used
        if left <= 0:
            msg = f"simulation budget exhausted ({self.budget} calls used); send your final answer"
            return protocol.format_observation({"error": msg, "sim_calls_left": 0}), 0.0, {
                "type": "budget_exhausted", "design": _design_view(design)}

        err = accel.validate_design(design, int(self.task["pe_budget"])) if isinstance(design, dict) \
            else "design must be a JSON object"
        if err:
            entry = {"design": _design_view(design), "error": err}
            self.sim_log.append(entry)
            return protocol.format_observation({"error": err, "sim_calls_left": left}), 0.0, dict(
                {"type": "invalid_simulate"}, **entry)

        try:
            out = accel.simulate(self.task["layer"], design["array_h"], design["array_w"],
                                 design["dataflow"], design["sram_split"])
        except Exception as exc:  # simulator crash: report, do not charge budget
            entry = {"design": _design_view(design), "error": f"simulator error: {exc}"}
            self.sim_log.append(entry)
            return protocol.format_observation({"error": entry["error"], "sim_calls_left": left}), 0.0, dict(
                {"type": "sim_error"}, **entry)

        self.sim_calls_used += 1
        entry = {"design": _design_view(design), "cycles": out.get("cycles"),
                 "utilization_pct": out.get("utilization_pct"), "stall_cycles": out.get("stall_cycles")}
        self.sim_log.append(entry)
        result = dict(entry, sim_calls_used=self.sim_calls_used, sim_calls_left=self.budget - self.sim_calls_used)
        return protocol.format_observation(result), 0.0, dict({"type": "simulate", "call": self.sim_calls_used}, **entry)
