"""Evaluation runner: drives Sim2GymEnv with a model or a baseline and writes results JSONL.

The model only ever sees the rendered prompt and env observations; the task's optimum is never
put into a message. Results record the optimum (for reports/the live demo) after grading.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sim2gym import models as models_mod
from sim2gym.baselines import BASELINES, get_baseline
from sim2gym.env import Sim2GymEnv

EventFn = Callable[[dict], None]
REASONING_KEEP_CHARS = 4000  # reasoning is recorded (truncated) for inspection, never sent back to the model

# Paid-model guard: Bedrock costs money, so by default a run is a tiny smoke test written to results/smoke/
# (not the leaderboard) unless allow_paid=True (CLI --allow-paid, API env SIM2GYM_ALLOW_PAID=1).
PAID_PREFIXES = ("bedrock:",)
PAID_GUARD_MAX_TASKS = 2
PAID_GUARD_MAX_TURNS = 2


def is_paid(model: str) -> bool:
    return model.startswith(PAID_PREFIXES)


def default_workers(model: str) -> int:
    return 4 if model.startswith("bedrock:") else 1  # openrouter free tier / local Ollama: 1


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def task_set_hash(tasks: list[dict]) -> str:
    canon = json.dumps(sorted(tasks, key=lambda t: t["id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def load_tasks(path) -> list[dict]:
    from sim2gym import tasks as tasks_mod
    return tasks_mod.load_tasks(path)


class LLMPolicy:
    """Chat loop over a model client. Holds the conversation for one episode."""
    is_baseline = False

    def __init__(self, model):
        self.model = model
        self.name = model.name
        self.last = None

    def settings(self) -> dict:
        return self.model.settings()

    def start(self, task: dict) -> None:
        self.messages = [{"role": "user", "content": task["prompt"]}]  # prompt only: no optimum
        self.first = True

    def act(self, obs: str, info: dict | None) -> str:
        if not self.first:
            self.messages.append({"role": "user", "content": obs})
        self.first = False
        self.last = self.model.chat(self.messages)
        self.messages.append({"role": "assistant", "content": self.last.text or "(empty reply)"})
        return self.last.text


def make_policy_factory(spec: str, **model_settings) -> tuple[Callable[[], object], bool]:
    """Returns (factory producing a fresh per-episode policy, is_baseline)."""
    if spec in BASELINES:
        seed = int(model_settings.get("seed", 0))
        return (lambda: get_baseline(spec, seed=seed)), True
    settings = {k: v for k, v in model_settings.items() if k in ("max_tokens", "timeout_s", "temperature") and v is not None}
    model = models_mod.get_model(spec, **settings)  # one client shared across threads (boto3 is thread-safe)
    return (lambda: LLMPolicy(model)), False


def run_episode(task: dict, policy, model_name: str, budget: int | None = None, tset_hash: str | None = None,
                on_event: EventFn | None = None, max_turns: int | None = None) -> dict:
    emit = on_event or (lambda e: None)
    env = Sim2GymEnv([task], budget=budget, max_turns=max_turns)
    t_wall = time.monotonic()
    prompt = env.reset(0)
    policy.start(env.task)
    emit({"type": "start", "task_id": task["id"], "model": model_name, "budget": env.budget,
          "max_turns": env.max_turns, "baseline": bool(policy.is_baseline), "prompt": prompt})
    transcript = [{"role": "user", "content": prompt}]
    obs, info, latency, error, infra_error, detail = prompt, None, 0.0, None, False, None
    usage: dict = {}
    while not env.done:
        text = policy.act(obs, info)
        last = getattr(policy, "last", None)
        if last is not None:
            latency += last.latency_s
            for k, v in (last.usage or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            if last.error:
                error, infra_error, detail = last.error, True, last.detail
                emit({"type": "model_error", "error": last.error, "detail": last.detail,
                      "latency_s": round(last.latency_s, 3)})
                break
        entry = {"role": "assistant", "content": text}
        if last is not None and last.reasoning:
            entry["reasoning_chars"] = len(last.reasoning)
            entry["reasoning"] = last.reasoning[:REASONING_KEEP_CHARS]
        if last is not None and last.stop_reason:
            entry["stop_reason"] = last.stop_reason
        transcript.append(entry)
        emit({"type": "model", "text": text, "latency_s": round(last.latency_s, 3) if last is not None else 0.0,
              "has_reasoning": bool(last is not None and last.reasoning)})
        obs, reward, done, info = env.step(text)
        transcript.append({"role": "user", "content": obs})
        ev = dict(info["event"])
        ev.update(type="env_" + ev["type"], sim_calls_used=info["sim_calls_used"], budget=info["budget"],
                  turns=info["turns"], observation=obs)
        emit(ev)

    grade = env.grade or {"score": 0.0, "solved": False, "feasible": False, "cycles": None, "error": error}
    opt = task.get("optimum") or {}
    result = {
        "task_id": task["id"], "model": model_name, "baseline": bool(policy.is_baseline),
        "kind": task.get("kind"), "sim": task.get("sim"), "sim_version": task.get("sim_version"),
        "task_set_hash": tset_hash, "layer_name": task.get("layer_name"), "pe_budget": task.get("pe_budget"),
        "score": float(grade.get("score", 0.0)), "solved": bool(grade.get("solved", False)),
        "feasible": bool(grade.get("feasible", False)), "answer": env.answer, "answer_cycles": grade.get("cycles"),
        "optimum_inputs": opt.get("inputs"), "optimum_cycles": opt.get("value"),
        "sim_calls_used": env.sim_calls_used, "budget": env.budget, "turns": env.turns,
        "retries_used": env.retries_used, "latency_s": round(latency, 3),
        "wall_s": round(time.monotonic() - t_wall, 3), "error": error or env.error, "error_detail": detail,
        "infra_error": infra_error,
        "max_turns": env.max_turns, "paid_guard": max_turns is not None and is_paid(model_name),
        "sim_log": env.sim_log, "transcript": transcript, "model_settings": policy.settings(), "usage": usage,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    emit({"type": "final", "result": {k: v for k, v in result.items() if k != "transcript"}})
    return result


def _read_done(path: Path, tset_hash: str) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("task_set_hash") == tset_hash and not r.get("infra_error"):
            done.add(r["task_id"])
    return done


def run(tasks: list[dict], model: str, results_dir="results", workers: int | None = None, budget: int | None = None,
        limit: int | None = None, task_ids: list[str] | None = None, resume: bool = True, out_path=None,
        on_result: Callable[[dict], None] | None = None, allow_paid: bool = False, **model_settings) -> dict:
    tset_hash = task_set_hash(tasks)
    factory, is_baseline = make_policy_factory(model, **model_settings)
    selected = [t for t in tasks if task_ids is None or t["id"] in set(task_ids)]
    guard = is_paid(model) and not allow_paid
    max_turns = PAID_GUARD_MAX_TURNS if guard else None
    if guard:
        limit = min(limit or PAID_GUARD_MAX_TASKS, PAID_GUARD_MAX_TASKS)
    if limit is not None:
        selected = selected[:limit]
    sub = Path(results_dir) / "smoke" if guard else Path(results_dir)
    out = Path(out_path) if out_path else sub / f"{safe_name(model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done(out, tset_hash) if resume else set()
    todo = [t for t in selected if t["id"] not in done]
    if workers is None:
        workers = default_workers(model)
    lock = threading.Lock()
    results: list[dict] = []
    stop = threading.Event()  # set on rate_limited so we don't keep hammering a capped free tier

    def one(task):
        if stop.is_set():
            return None
        r = run_episode(task, factory(), model, budget=budget, tset_hash=tset_hash, max_turns=max_turns)
        if r.get("error") == "rate_limited":
            stop.set()
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(r) + "\n")
            results.append(r)
            if on_result:
                on_result(r)
        return r

    if workers <= 1:
        for t in todo:
            one(t)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(one, t) for t in todo]):
                fut.result()
    n = len(results)
    return {"model": model, "baseline": is_baseline, "out": str(out), "task_set_hash": tset_hash,
            "n_selected": len(selected), "n_skipped": len(selected) - len(todo), "n_run": n,
            "mean_score": (sum(r["score"] for r in results) / n) if n else None,
            "n_errors": sum(1 for r in results if r["infra_error"]), "rate_limited": stop.is_set(),
            "paid_guard": guard, "max_turns": max_turns}
