"""sim2gym HTTP API + realtime demo.

Local demo server: bind 127.0.0.1, single uvicorn worker, NO auth (not for public multi-user use).
Sessions, task sets and runs live in memory and are lost on restart; run results are also written to
``results/runs/<run_id>.jsonl`` and live-demo episodes to ``results/live/<model>.jsonl`` (neither is
picked up by the leaderboard, which reads only ``results/*.jsonl`` written by ``sim2gym run``).

Config (server-side env only): SIM2GYM_TASKS (default ./tasks.jsonl), SIM2GYM_RESULTS (default ./results),
SIM2GYM_OPENAI_MODELS (comma list of extra ``openai:<model>`` names to allow), plus the model env vars in
models.py. Clients can only choose models from the allowlist and never pass paths, URLs or code.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from sim2gym import __version__, insights, report, runner
from sim2gym import tasks as tasks_mod
from sim2gym.baselines import BASELINES
from sim2gym.env import Sim2GymEnv
from sim2gym import models as models_mod
from sim2gym.models import BEDROCK_MODELS
from sim2gym.packs import accel

STATIC = Path(__file__).parent / "static"
ID_RE = r"^[A-Za-z0-9_-]{1,40}$"
TASK_ID_RE = r"^[A-Za-z0-9_./-]{1,160}$"
MAX_SESSIONS = 200
MAX_TASK_SETS = 20
MAX_LIVE = 3
TRIM = 2000


def allow_paid() -> bool:
    return os.environ.get("SIM2GYM_ALLOW_PAID") == "1"


def allowed_models() -> list[dict]:
    """Server-side allowlist, in dropdown order: OpenRouter free (discovered, cached 10 min), local
    OpenAI-compatible (SIM2GYM_OPENAI_MODELS), baselines, then Bedrock (paid, short tests)."""
    out = [{"id": f"openrouter:{m['id']}", "label": m["name"], "group": "free", "baseline": False}
           for m in models_mod.openrouter_free_models()]
    for m in filter(None, (x.strip() for x in os.environ.get("SIM2GYM_OPENAI_MODELS", "").split(","))):
        out.append({"id": f"openai:{m}", "label": f"{m} (local)", "group": "local", "baseline": False})
    out += [{"id": b, "label": f"{b} (baseline)", "group": "baseline", "baseline": True} for b in BASELINES]
    out += [{"id": f"bedrock:{m}", "label": m, "group": "bedrock", "baseline": False} for m in BEDROCK_MODELS]
    return out


def default_model() -> str:
    override = os.environ.get("SIM2GYM_DEFAULT_MODEL", "").strip()
    if override:  # server-side choice, e.g. bedrock:qwen.qwen3-32b-v1:0 when the free quota is used up
        return override
    return models_mod.default_free_model(models_mod.openrouter_free_models()) or models_mod.FALLBACK_DEMO_MODEL


def _check_model(model: str) -> str:
    static = set(BASELINES) | {f"bedrock:{m}" for m in BEDROCK_MODELS}
    if model not in static and model not in {m["id"] for m in allowed_models()}:
        raise HTTPException(400, f"model not in server allowlist: {model[:80]!r}")
    return model


# ------------------------------------------------------------------ schemas
class SimulateReq(BaseModel):
    workload: str = Field(..., max_length=80)
    pe_budget: int
    design: dict[str, Any]


class TaskSetReq(BaseModel):
    seed: int = Field(0, ge=0, le=1_000_000)
    budget: int = Field(6, ge=0, le=20)
    workloads: Optional[list[str]] = Field(None, max_length=50)
    pe_budgets: Optional[list[int]] = Field(None, max_length=10)


class SessionReq(BaseModel):
    task_set_id: str = Field("default", pattern=ID_RE)
    task_id: str = Field(..., pattern=TASK_ID_RE)


class StepReq(BaseModel):
    text: str = Field(..., max_length=20_000)


class RunReq(BaseModel):
    task_set_id: str = Field("default", pattern=ID_RE)
    model: str = Field(..., max_length=120)
    limit: int = Field(5, ge=1, le=50)
    task_ids: Optional[list[str]] = Field(None, max_length=50)
    workers: int = Field(2, ge=1, le=8)


# ------------------------------------------------------------------ app
def create_app(tasks_path: str | os.PathLike | None = None, results_dir: str | os.PathLike | None = None) -> FastAPI:
    tasks_path = Path(tasks_path or os.environ.get("SIM2GYM_TASKS", "tasks.jsonl"))
    results_dir = Path(results_dir or os.environ.get("SIM2GYM_RESULTS", "results"))
    app = FastAPI(title="sim2gym", version=__version__,
                  description="AccelGym on real SCALE-Sim: budgeted black-box design tasks. Local demo, no auth.")
    lock = threading.Lock()
    live_sem = threading.BoundedSemaphore(MAX_LIVE)
    state: dict[str, Any] = {"task_sets": {}, "sessions": {}, "runs": {}, "default_mtime": None}

    def task_sets() -> dict:
        """Task sets by id; 'default' is (re)loaded from the server's tasks file when it changes."""
        with lock:
            try:
                mtime = tasks_path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None and mtime != state["default_mtime"]:
                ts = tasks_mod.load_tasks(tasks_path)
                state["task_sets"]["default"] = {"tasks": ts, "hash": runner.task_set_hash(ts), "source": "file"}
                state["default_mtime"] = mtime
            return state["task_sets"]

    def get_set(set_id: str) -> dict:
        ts = task_sets().get(set_id)
        if ts is None:
            raise HTTPException(404, "unknown task set")
        return ts

    def get_task(set_id: str, task_id: str) -> dict:
        for t in get_set(set_id)["tasks"]:
            if t["id"] == task_id:
                return t
        raise HTTPException(404, "unknown task id")

    # ---------------------------------------------------------- basic
    @app.get("/health")
    def health():
        sets = task_sets()
        return {"status": "ok", "version": __version__, "simulator": accel.SIM_NAME,
                "scalesim_version": accel.SCALESIM_VERSION, "default_tasks": len(sets.get("default", {}).get("tasks", [])),
                "models_checked": False,
                "note": "model endpoints are not probed by /health; a run reports per-call errors"}

    @app.get("/simulators")
    def simulators():
        return [{"name": accel.SIM_NAME, "engine": f"SCALE-Sim {accel.SCALESIM_VERSION}",
                 "objective": "minimize cycles", "units": {"cycles": "clock cycles", "utilization_pct": "%"},
                 "design_space": {"array_h": accel.ARRAY_DIMS, "array_w": accel.ARRAY_DIMS,
                                  "dataflow": accel.DATAFLOWS, "sram_split": accel.SRAM_SPLITS},
                 "constraint": "array_h * array_w <= pe_budget", "pe_budgets": accel.PE_BUDGETS,
                 "workloads": [{"name": n, "kind": w["kind"], "source": w.get("source"),
                                "description": accel.describe_layer(w)} for n, w in accel.WORKLOADS.items()]}]

    @app.post("/simulators/{name}/simulate")
    def simulate(name: str, req: SimulateReq):
        if name != accel.SIM_NAME:
            raise HTTPException(404, "unknown simulator")
        if req.workload not in accel.WORKLOADS:
            raise HTTPException(422, "unknown workload")
        if req.pe_budget not in accel.PE_BUDGETS:
            raise HTTPException(422, f"pe_budget must be one of {accel.PE_BUDGETS}")
        err = accel.validate_design(req.design, req.pe_budget)
        if err:
            raise HTTPException(422, err)
        d = req.design
        out = accel.simulate(accel.WORKLOADS[req.workload], d["array_h"], d["array_w"], d["dataflow"], d["sram_split"])
        return {"workload": req.workload, "design": d, **out}

    @app.get("/models")
    def models():
        return {"models": allowed_models(), "default": default_model(), "allow_paid": allow_paid(),
                "paid_guard": {"max_tasks": runner.PAID_GUARD_MAX_TASKS, "max_turns": runner.PAID_GUARD_MAX_TURNS},
                "discovery_error": models_mod.free_models_error()}

    # ---------------------------------------------------------- tasks
    @app.post("/tasks")
    def create_tasks(req: TaskSetReq):
        if req.workloads is not None and any(w not in accel.WORKLOADS for w in req.workloads):
            raise HTTPException(422, "unknown workload")
        if req.pe_budgets is not None and any(p not in accel.PE_BUDGETS for p in req.pe_budgets):
            raise HTTPException(422, f"pe_budgets must be from {accel.PE_BUDGETS}")
        ts = tasks_mod.build_tasks(seed=req.seed, budget=req.budget, workloads=req.workloads, pe_budgets=req.pe_budgets)
        if not ts:
            raise HTTPException(422, "no tasks generated")
        set_id = "ts_" + uuid.uuid4().hex[:10]
        with lock:
            sets = state["task_sets"]
            user_sets = [k for k in sets if k != "default"]
            if len(user_sets) >= MAX_TASK_SETS:
                sets.pop(user_sets[0])
            sets[set_id] = {"tasks": ts, "hash": runner.task_set_hash(ts), "source": "api"}
        return {"id": set_id, "n": len(ts), "task_set_hash": runner.task_set_hash(ts),
                "tasks": [tasks_mod.public_task(t) for t in ts]}

    @app.get("/tasks")
    def list_task_sets():
        return [{"id": k, "n": len(v["tasks"]), "task_set_hash": v["hash"], "source": v["source"]}
                for k, v in task_sets().items()]

    @app.get("/tasks/{set_id}")
    def get_task_set(set_id: str):
        if not __import__("re").match(ID_RE, set_id):
            raise HTTPException(422, "bad task set id")
        s = get_set(set_id)
        return {"id": set_id, "n": len(s["tasks"]), "task_set_hash": s["hash"],
                "tasks": [tasks_mod.public_task(t) for t in s["tasks"]]}

    # ---------------------------------------------------------- sessions
    @app.post("/sessions")
    def create_session(req: SessionReq):
        task = get_task(req.task_set_id, req.task_id)
        env = Sim2GymEnv([task], budget=None)
        prompt = env.reset(0)
        sid = "s_" + uuid.uuid4().hex[:12]
        with lock:
            sess = state["sessions"]
            if len(sess) >= MAX_SESSIONS:
                sess.pop(next(iter(sess)))
            sess[sid] = {"env": env, "lock": threading.Lock(), "task": task, "task_set_id": req.task_set_id}
        return {"session_id": sid, "task_id": task["id"], "prompt": prompt, "budget": env.budget,
                "max_turns": env.max_turns}

    def _session(sid: str) -> dict:
        with lock:
            s = state["sessions"].get(sid)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    def _session_view(s: dict, info: dict | None = None) -> dict:
        env = s["env"]
        view = {"task_id": s["task"]["id"], "done": env.done, "sim_calls_used": env.sim_calls_used,
                "budget": env.budget, "turns": env.turns, "max_turns": env.max_turns, "sim_log": env.sim_log,
                "answer": env.answer, "grade": env.grade, "error": env.error}
        if env.done:  # reveal the reference only after the episode is over
            view["optimum"] = s["task"].get("optimum")
        return view

    @app.get("/sessions/{sid}")
    def get_session(sid: str):
        s = _session(sid)
        with s["lock"]:
            return _session_view(s)

    @app.post("/sessions/{sid}/step")
    def step(sid: str, req: StepReq):
        s = _session(sid)
        with s["lock"]:
            if s["env"].done:
                raise HTTPException(409, "session is terminal")
            obs, reward, done, info = s["env"].step(req.text)
            return {"observation": obs, "reward": reward, "event": info["event"], **_session_view(s)}

    # ---------------------------------------------------------- runs
    @app.post("/runs")
    def create_run(req: RunReq):
        model = _check_model(req.model)
        s = get_set(req.task_set_id)
        if req.task_ids is not None:
            known = {t["id"] for t in s["tasks"]}
            if any(t not in known for t in req.task_ids):
                raise HTTPException(422, "unknown task id")
        run_id = "r_" + uuid.uuid4().hex[:10]
        out = results_dir / "runs" / f"{run_id}.jsonl"
        rec = {"id": run_id, "model": model, "task_set_id": req.task_set_id, "status": "running", "results": [],
               "summary": None, "error": None, "started": time.time()}
        with lock:
            state["runs"][run_id] = rec

        def on_result(r):
            with lock:
                rec["results"].append({k: v for k, v in r.items() if k != "transcript"})

        def work():
            try:
                summ = runner.run(s["tasks"], model, workers=min(req.workers, runner.default_workers(model)),
                                  limit=req.limit, task_ids=req.task_ids, resume=False, out_path=out,
                                  on_result=on_result, allow_paid=allow_paid())
                with lock:
                    rec.update(status="done", summary=summ)
            except Exception as exc:
                with lock:
                    rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:300])

        threading.Thread(target=work, daemon=True).start()
        return {"id": run_id, "status": "running"}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        with lock:
            rec = state["runs"].get(run_id)
            if rec is None:
                raise HTTPException(404, "unknown run")
            return json.loads(json.dumps(rec, default=str))

    @app.get("/leaderboard")
    def leaderboard():
        default = task_sets().get("default", {}).get("tasks")
        lb = report.build_leaderboard(results_dir, tasks=default) if results_dir.exists() else {
            "rows": [], "warnings": [], "task_set_hashes": []}
        return {**lb, "markdown": report.to_markdown(lb)}

    @app.get("/insights")
    def get_insights():
        """Plain-language analysis of the saved results on the default task set (complete runs only)."""
        with lock:
            ts = list(state["task_sets"].get("default", {}).get("tasks", []))
        return insights.build_insights(ts, results_dir)

    # ---------------------------------------------------------- demo
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/live")
    def live(task_id: str = Query(..., pattern=TASK_ID_RE), model: str = Query(..., max_length=120),
             task_set_id: str = Query("default", pattern=ID_RE)):
        model = _check_model(model)
        task = get_task(task_set_id, task_id)
        if not live_sem.acquire(blocking=False):
            raise HTTPException(429, "too many live episodes running; try again shortly")
        q: queue.Queue = queue.Queue()
        tset_hash = get_set(task_set_id)["hash"]

        def trim(ev: dict) -> dict:
            ev = dict(ev)
            for k in ("text", "observation"):
                if isinstance(ev.get(k), str) and len(ev[k]) > TRIM:
                    ev[k] = ev[k][:TRIM] + " ...[trimmed]"
            return ev

        def work():
            try:
                pub = tasks_mod.public_task(task)
                q.put({"type": "task", "task": {k: pub.get(k) for k in ("id", "layer_name", "pe_budget", "space",
                                                                         "budget")},
                       "layer": accel.describe_layer(task["layer"])})
                factory, _ = runner.make_policy_factory(model)
                guard = runner.is_paid(model) and not allow_paid()
                r = runner.run_episode(task, factory(), model, tset_hash=tset_hash, on_event=lambda e: q.put(trim(e)),
                                       max_turns=runner.PAID_GUARD_MAX_TURNS if guard else None)
                try:
                    p = results_dir / "live" / f"{runner.safe_name(model)}.jsonl"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with lock, p.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                except OSError:
                    pass
                q.put({"type": "optimum", "inputs": task["optimum"]["inputs"], "cycles": task["optimum"]["value"],
                       "method": task["optimum"].get("method")})
            except Exception as exc:
                q.put({"type": "server_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            finally:
                q.put(None)
                live_sem.release()

        threading.Thread(target=work, daemon=True).start()

        def stream():
            yield "retry: 100000000\n\n"
            while True:
                try:
                    ev = q.get(timeout=10)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if ev is None:
                    yield "event: end\ndata: {}\n\n"
                    return
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
