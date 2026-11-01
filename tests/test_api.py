"""HTTP API integration: task set -> session -> simulate -> answer -> result; validation; SSE /live."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from sim2gym import api, models, tasks
from sim2gym.packs import accel

from test_env import WL, fake_simulate


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    monkeypatch.setattr(models, "openrouter_free_models",
                        lambda force=False: [{"id": "qwen/qwen3-8b:free", "name": "Qwen3 8B (free)"}])
    monkeypatch.delenv("SIM2GYM_ALLOW_PAID", raising=False)
    tp = tmp_path / "tasks.jsonl"
    tasks.save_tasks(tp, tasks.build_tasks(seed=0, workloads=[WL], pe_budgets=[256, 1024]))
    return TestClient(api.create_app(tasks_path=tp, results_dir=tmp_path / "results"))


def D(**kw):
    d = {"array_h": 8, "array_w": 16, "dataflow": "os", "sram_split": "balanced"}
    d.update(kw)
    return d


def test_health_simulators_models_index(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["default_tasks"] == 2 and h["models_checked"] is False
    s = client.get("/simulators").json()[0]
    assert s["name"] == "accel" and WL in [w["name"] for w in s["workloads"]]
    m = client.get("/models").json()
    groups = [x["group"] for x in m["models"]]
    assert groups[0] == "free" and groups.index("baseline") < groups.index("bedrock")
    assert m["default"] == "openrouter:qwen/qwen3-8b:free"
    r = client.get("/")
    assert r.status_code == 200 and "EventSource" in r.text


def test_full_session_flow(client):
    created = client.post("/tasks", json={"seed": 1, "budget": 3, "workloads": [WL], "pe_budgets": [256]})
    assert created.status_code == 200
    body = created.json()
    assert "optimum" not in json.dumps(body) and body["n"] == 1
    set_id, task_id = body["id"], body["tasks"][0]["id"]
    pub = client.get(f"/tasks/{set_id}").json()
    assert "optimum" not in json.dumps(pub) and "exhaustive" not in json.dumps(pub)

    sess = client.post("/sessions", json={"task_set_id": set_id, "task_id": task_id}).json()
    assert sess["budget"] == 3 and sess["max_turns"] == 5 and "optimum" not in json.dumps(sess)
    sid = sess["session_id"]
    st = client.post(f"/sessions/{sid}/step", json={"text": json.dumps({"simulate": D()})}).json()
    assert st["done"] is False and st["sim_calls_used"] == 1 and "cycles" in json.loads(st["observation"])
    assert "optimum" not in st
    st = client.post(f"/sessions/{sid}/step", json={"text": json.dumps({"answer": D()})}).json()
    assert st["done"] and st["grade"]["feasible"] and 0 < st["reward"] <= 1 and st["optimum"]["method"] == "exhaustive"
    got = client.get(f"/sessions/{sid}").json()
    assert got["grade"] == st["grade"] and got["answer"] == D()
    assert client.post(f"/sessions/{sid}/step", json={"text": "{}"}).status_code == 409

    sim = client.post("/simulators/accel/simulate", json={"workload": WL, "pe_budget": 256, "design": D()}).json()
    assert sim["cycles"] == st["grade"]["cycles"]


@pytest.mark.parametrize("method,url,payload,code", [
    ("post", "/simulators/accel/simulate", {"workload": WL, "pe_budget": 256, "design": D(array_h=64, array_w=64)}, 422),
    ("post", "/simulators/accel/simulate", {"workload": "../etc/passwd", "pe_budget": 256, "design": D()}, 422),
    ("post", "/simulators/accel/simulate", {"workload": WL, "pe_budget": 300, "design": D()}, 422),
    ("post", "/simulators/nope/simulate", {"workload": WL, "pe_budget": 256, "design": D()}, 404),
    ("post", "/tasks", {"seed": -1}, 422),
    ("post", "/tasks", {"budget": 999}, 422),
    ("post", "/tasks", {"workloads": ["nope"]}, 422),
    ("post", "/tasks", {"pe_budgets": [123]}, 422),
    ("post", "/sessions", {"task_set_id": "default", "task_id": "accel/design/nope/pe1"}, 404),
    ("post", "/sessions", {"task_set_id": "../x", "task_id": "a"}, 422),
    ("post", "/sessions/s_missing/step", {"text": "x"}, 404),
    ("post", "/runs", {"model": "openai:http://evil", "task_set_id": "default"}, 400),
    ("post", "/runs", {"model": "oracle", "task_set_id": "default", "limit": 1000}, 422),
    ("get", "/tasks/missing", None, 404),
    ("get", "/runs/r_missing", None, 404),
    ("get", "/live?task_id=x&model=oracle", None, 404),
])
def test_invalid_requests(client, method, url, payload, code):
    r = client.post(url, json=payload) if method == "post" else client.get(url)
    assert r.status_code == code, r.text


def test_step_text_too_long(client):
    tid = client.get("/tasks/default").json()["tasks"][0]["id"]
    sid = client.post("/sessions", json={"task_id": tid}).json()["session_id"]
    assert client.post(f"/sessions/{sid}/step", json={"text": "x" * 20001}).status_code == 422


def _sse(client, url):
    events = []
    with client.stream("GET", url) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event: end"):
                break
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_live_sse_oracle(client):
    tid = client.get("/tasks/default").json()["tasks"][0]["id"]
    ev = _sse(client, f"/live?task_id={tid}&model=oracle")
    types = [e["type"] for e in ev]
    assert types[:2] == ["task", "start"] and "model" in types and "env_answer" in types
    final = next(e for e in ev if e["type"] == "final")["result"]
    assert final["score"] == 1.0 and final["baseline"] is True
    opt = next(e for e in ev if e["type"] == "optimum")
    assert opt["cycles"] == final["answer_cycles"]
    assert "optimum" not in json.dumps(ev[0])  # task event is public view


def test_live_sse_probe_streams_simulations(client):
    tid = client.get("/tasks/default").json()["tasks"][0]["id"]
    ev = _sse(client, f"/live?task_id={tid}&model=6-probe")
    sims = [e for e in ev if e["type"] == "env_simulate"]
    assert len(sims) == 6 and all("cycles" in s and "utilization_pct" in s for s in sims)
    assert [s["sim_calls_used"] for s in sims] == list(range(1, 7))


def test_live_rejects_unlisted_model(client):
    tid = client.get("/tasks/default").json()["tasks"][0]["id"]
    assert client.get(f"/live?task_id={tid}&model=openai:gpt-4").status_code == 400


def test_runs_and_leaderboard(client):
    run = client.post("/runs", json={"model": "6-probe", "task_set_id": "default", "limit": 2}).json()
    for _ in range(100):
        rec = client.get(f"/runs/{run['id']}").json()
        if rec["status"] != "running":
            break
        time.sleep(0.05)
    assert rec["status"] == "done" and len(rec["results"]) == 2 and rec["summary"]["baseline"]
    assert all("transcript" not in r for r in rec["results"])
    lb = client.get("/leaderboard").json()
    assert "rows" in lb and "markdown" in lb
