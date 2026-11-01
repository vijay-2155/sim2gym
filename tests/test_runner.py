"""PROTOCOL TESTS ONLY: the runner and model clients against a fake local OpenAI-compatible HTTP server.

These check the chat loop, experiment -> answer, timeouts, failures, 429 handling, OpenRouter discovery
and quota parsing without any credentials. They are NOT a real open-model evaluation.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sim2gym import models, runner, tasks
from sim2gym.packs import accel

from test_env import WL, fake_simulate


class Fake:
    def __init__(self):
        self.script = []      # list of ("reply", text) | ("status", code, headers) | ("sleep", s, text)
        self.requests = []    # (path, headers, body)
        self.models = {"data": []}
        self.key = {"data": {}}


def _server():
    fake = Fake()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, headers=None):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def do_GET(self):
            fake.requests.append((self.path, dict(self.headers), None))
            if self.path.endswith("/models"):
                return self._send(200, fake.models)
            if self.path.endswith("/key"):
                return self._send(200, fake.key)
            self._send(404, {"error": "nope"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append((self.path, dict(self.headers), body))
            step = fake.script.pop(0) if fake.script else ("status", 500, {})
            if step[0] == "sleep":
                time.sleep(step[1])
                step = ("reply", step[2])
            if step[0] == "status":
                return self._send(step[1], {"error": {"message": f"fake {step[1]}"}}, step[2])
            self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                             "choices": [{"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": step[1]}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, fake


@pytest.fixture
def fake_server(monkeypatch):
    srv, fake = _server()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    monkeypatch.setenv("SIM2GYM_BASE_URL", url)
    monkeypatch.setenv("SIM2GYM_OPENROUTER_BASE_URL", url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-key")
    models._free_cache.update(t=0.0, models=None, error=None)
    yield fake
    srv.shutdown()


@pytest.fixture
def task_list(monkeypatch):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    return tasks.build_tasks(seed=0, budget=6, workloads=[WL], pe_budgets=[256, 1024, 4096])


def D(**kw):
    d = {"array_h": 8, "array_w": 8, "dataflow": "ws", "sram_split": "balanced"}
    d.update(kw)
    return d


def test_experiment_then_answer(fake_server, task_list):
    t = task_list[0]
    fake_server.script = [("reply", "Try a probe.\n" + json.dumps({"simulate": D()})),
                          ("reply", "```json\n" + json.dumps({"answer": t["optimum"]["inputs"]}) + "\n```")]
    pol = runner.LLMPolicy(models.get_model("openai:fake-model", timeout_s=5))
    r = runner.run_episode(t, pol, "openai:fake-model", tset_hash="h")
    assert r["score"] == 1.0 and r["sim_calls_used"] == 1 and r["turns"] == 2 and r["error"] is None
    assert [m["role"] for m in r["transcript"]] == ["user", "assistant", "user", "assistant", "user"]
    assert r["model_settings"]["temperature"] == 0.0 and r["usage"]["total_tokens"] == 30
    bodies = [b for p, h, b in fake_server.requests if b]
    assert all(b["temperature"] == 0 for b in bodies)
    # the optimum is never sent to the model
    sent = json.dumps(bodies)
    assert "optimum" not in sent and "exhaustive" not in sent
    assert len(bodies[1]["messages"]) == 3 and '"cycles"' in bodies[1]["messages"][2]["content"]


def test_think_blocks_stripped(fake_server, task_list):
    t = task_list[0]
    fake_server.script = [("reply", "<think>{\"answer\": {\"array_h\": 64}}</think>\n" + json.dumps({"answer": D()}))]
    r = runner.run_episode(t, runner.LLMPolicy(models.get_model("openai:m")), "openai:m")
    assert r["feasible"] and r["transcript"][1]["reasoning"]


def test_timeout_is_clean_failure(fake_server, task_list):
    fake_server.script = [("sleep", 1.5, json.dumps({"answer": D()}))]
    m = models.OpenAICompatModel("slow", timeout_s=0.3, retries=0)
    r = runner.run_episode(task_list[0], runner.LLMPolicy(m), m.name)
    assert r["infra_error"] and r["score"] == 0.0 and r["error"] == "provider_error" and "Timeout" in r["error_detail"]
    assert r["latency_s"] < 1.5


def test_server_failure_is_clean_failure(fake_server, task_list):
    fake_server.script = [("status", 400, {})]
    m = models.OpenAICompatModel("bad", timeout_s=5, retries=0)
    r = runner.run_episode(task_list[0], runner.LLMPolicy(m), m.name)
    assert r["infra_error"] and r["score"] == 0.0 and r["error"].startswith("openai error")


def test_openrouter_429_then_success(fake_server, task_list):
    t = task_list[0]
    fake_server.script = [("status", 429, {"Retry-After": "0"}),
                          ("reply", json.dumps({"answer": t["optimum"]["inputs"]}))]
    m = models.get_model("openrouter:qwen/qwen3-8b:free")
    r = runner.run_episode(t, runner.LLMPolicy(m), m.name)
    assert r["score"] == 1.0 and r["error"] is None
    posts = [(p, h) for p, h, b in fake_server.requests if b]
    assert len(posts) == 2
    h = {k.lower(): v for k, v in posts[-1][1].items()}
    assert h["authorization"] == "Bearer test-secret-key" and h["x-title"] == "sim2gym" and "http-referer" in h
    assert "test-secret-key" not in json.dumps(r)  # key never saved


def test_openrouter_persistent_429_rate_limited_and_run_stops(fake_server, task_list, tmp_path):
    fake_server.script = [("status", 429, {"Retry-After": "0"})] * 6 + [("reply", json.dumps({"answer": D()}))] * 5
    s = runner.run(task_list, "openrouter:qwen/qwen3-8b:free", results_dir=tmp_path)
    assert s["rate_limited"] and s["n_run"] == 1 and s["n_errors"] == 1
    r = json.loads((tmp_path / "openrouter_qwen_qwen3-8b_free.jsonl").read_text().splitlines()[0])
    assert r["error"] == "rate_limited" and r["infra_error"]
    assert len([1 for p, h, b in fake_server.requests if b]) == 6  # 1 try + 5 retries, then stop


def test_openrouter_long_retry_after_fails_fast(fake_server, task_list):
    fake_server.script = [("status", 429, {"Retry-After": "3600"})]
    m = models.get_model("openrouter:x:free")
    t0 = time.monotonic()
    assert m.chat([{"role": "user", "content": "hi"}]).error == "rate_limited"
    assert time.monotonic() - t0 < 5


def test_openrouter_missing_key(monkeypatch, task_list):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    m = models.get_model("openrouter:x:free")
    assert "OPENROUTER_API_KEY" in m.chat([{"role": "user", "content": "hi"}]).error
    assert models.openrouter_quota()["error"].startswith("OPENROUTER_API_KEY")


def test_free_model_discovery_and_quota(fake_server):
    fake_server.models = {"data": [
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": {"prompt": "0.1", "completion": "0.1"}},
        {"id": "qwen/qwen3-14b", "name": "Qwen3 14B", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        {"id": "google/gemma-3-4b-it:free", "pricing": {"prompt": "0", "completion": "0"}}]}
    free = models.openrouter_free_models(force=True)
    assert [m["id"] for m in free] == ["google/gemma-3-4b-it:free", "meta-llama/llama-3.3-70b-instruct:free",
                                       "qwen/qwen3-14b"]
    assert models.default_free_model(free) == "openrouter:qwen/qwen3-14b"
    fake_server.models = {"data": []}
    assert len(models.openrouter_free_models()) == 3  # cached
    fake_server.key = {"data": {"label": "sk-or-v1-abc...xyz", "limit": None, "usage": 0.0, "is_free_tier": True,
                                "rate_limit": {"requests": 20, "interval": "10s"}}}
    q = models.openrouter_quota()
    assert q == {"limit": None, "usage": 0.0, "is_free_tier": True, "rate_limit": {"requests": 20, "interval": "10s"}}


def test_baseline_run_resume_and_hash(task_list, tmp_path):
    s1 = runner.run(task_list, "oracle", results_dir=tmp_path)
    assert s1["n_run"] == 3 and s1["mean_score"] == 1.0
    s2 = runner.run(task_list, "oracle", results_dir=tmp_path)
    assert s2["n_run"] == 0 and s2["n_skipped"] == 3
    rows = [json.loads(x) for x in (tmp_path / "oracle.jsonl").read_text().splitlines()]
    assert all(r["baseline"] and r["task_set_hash"] == runner.task_set_hash(task_list) for r in rows)
    for b in ("random", "max-square-ws", "6-probe"):
        s = runner.run(task_list, b, results_dir=tmp_path)
        assert s["n_run"] == 3 and 0.0 <= s["mean_score"] <= 1.0
    probe = [json.loads(x) for x in (tmp_path / "6-probe.jsonl").read_text().splitlines()]
    assert all(r["sim_calls_used"] == 6 and r["feasible"] for r in probe)
    # 6-probe answers its best probe
    assert all(r["answer_cycles"] == min(e["cycles"] for e in r["sim_log"]) for r in probe)


def test_bedrock_paid_guard(task_list, tmp_path, monkeypatch):
    calls = []

    def fake_chat(self, messages):
        calls.append(messages)
        return models.ChatResult(json.dumps({"simulate": D()}), 0.01)

    monkeypatch.setattr(models.BedrockModel, "chat", fake_chat)
    s = runner.run(task_list, "bedrock:qwen.qwen3-32b-v1:0", results_dir=tmp_path)
    assert s["paid_guard"] and s["n_run"] == 2 and s["max_turns"] == 2
    rows = [json.loads(x) for x in (tmp_path / "smoke" / "bedrock_qwen.qwen3-32b-v1_0.jsonl").read_text().splitlines()]
    assert all(r["turns"] == 2 and r["paid_guard"] for r in rows) and len(calls) == 4
    assert not (tmp_path / "bedrock_qwen.qwen3-32b-v1_0.jsonl").exists()
    s = runner.run(task_list, "bedrock:qwen.qwen3-32b-v1:0", results_dir=tmp_path, allow_paid=True, limit=1)
    assert not s["paid_guard"] and s["n_run"] == 1


def test_bedrock_text_and_reasoning_blocks():
    class C:
        def converse(self, **kw):
            assert kw["inferenceConfig"]["temperature"] == 0.0 and kw["inferenceConfig"]["maxTokens"] == 6144
            return {"output": {"message": {"content": [
                {"reasoningContent": {"reasoningText": {"text": '{"answer": {"array_h": 64}}'}}},
                {"text": 'ok {"answer": {"array_h": 8}}'}]}}, "usage": {"inputTokens": 3}, "stopReason": "end_turn"}

    m = models.BedrockModel("openai.gpt-oss-20b-1:0", client=C())
    r = m.chat([{"role": "user", "content": "hi"}])
    assert r.text == 'ok {"answer": {"array_h": 8}}' and "64" in r.reasoning and r.error is None


def test_bedrock_throttle_retry_then_error():
    from botocore.exceptions import ClientError

    class C:
        n = 0

        def converse(self, **kw):
            C.n += 1
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "bad"}}, "Converse")

    r = models.BedrockModel("google.gemma-3-4b-it", client=C(), retries=3).chat([{"role": "user", "content": "x"}])
    assert C.n == 1 and "ValidationException" in r.error  # non-retryable: no retries


def test_dotenv_parser(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# c\nexport SIM2GYM_T1="a b"\nSIM2GYM_T2=\'x\'\nSIM2GYM_T3=\nSIM2GYM_T1_KEEP=1\n')
    monkeypatch.setenv("SIM2GYM_T1_KEEP", "orig")
    monkeypatch.delenv("SIM2GYM_T1", raising=False)
    monkeypatch.delenv("SIM2GYM_T2", raising=False)
    models.load_dotenv(p)
    import os
    assert os.environ["SIM2GYM_T1"] == "a b" and os.environ["SIM2GYM_T2"] == "x"
    assert os.environ["SIM2GYM_T1_KEEP"] == "orig" and "SIM2GYM_T3" not in os.environ
    monkeypatch.delenv("SIM2GYM_T1")
    monkeypatch.delenv("SIM2GYM_T2")


def test_openrouter_defaults():
    m = models.get_model("openrouter:nvidia/nemotron-3.5-super-vl:free")
    assert m.max_tokens == 12000 and m.temperature == 0.0 and m.retries == 5


def test_openrouter_503_then_success_and_provider_error(fake_server, task_list, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    fake_server.script = [("status", 503, {}), ("reply", json.dumps({"answer": D()}))]
    m = models.get_model("openrouter:recursive/flint-v1.0:free")
    assert m.chat([{"role": "user", "content": "x"}]).error is None
    fake_server.script = [("status", 503, {})] * 6
    r = m.chat([{"role": "user", "content": "x"}])
    assert r.error == "provider_error" and "503" in r.detail


def test_openrouter_choices_none_is_provider_error(fake_server, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    fake_server.script = []  # use a custom handler: 200 with an error body and no choices

    class M(models.OpenRouterModel):
        pass

    m = M("x:free")
    import openai

    class FakeResp:
        choices = None
        model_extra = {"error": {"message": "Provider returned error", "code": 429}}

    class FakeClient:
        n = 0

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    FakeClient.n += 1
                    return FakeResp()

    m._client = FakeClient()
    r = m.chat([{"role": "user", "content": "x"}])
    assert r.error == "provider_error" and "Provider returned error" in r.detail and 2 <= FakeClient.n <= 6


def test_openrouter_total_backoff_cap(fake_server, monkeypatch):
    slept = []
    monkeypatch.setattr(models.time, "sleep", lambda s: slept.append(s))
    fake_server.script = [("status", 429, {"Retry-After": "25"})] * 6
    r = models.get_model("openrouter:google/gemma-4-31b-it:free").chat([{"role": "user", "content": "x"}])
    assert r.error == "rate_limited" and sum(slept) <= 60 and len(slept) == 2


def test_reasoning_recorded_not_fed_back(fake_server, task_list, monkeypatch):
    class R(models.OpenAICompatModel):
        def _result(self, resp, latency):
            res = super()._result(resp, latency)
            res.reasoning = "r" * 9000
            return res

    fake_server.script = [("reply", json.dumps({"simulate": D()})), ("reply", json.dumps({"answer": D()}))]
    m = R("m")
    r = runner.run_episode(task_list[0], runner.LLMPolicy(m), m.name)
    a = r["transcript"][1]
    assert a["reasoning_chars"] == 9000 and len(a["reasoning"]) == runner.REASONING_KEEP_CHARS
    second = [b for p, h, b in fake_server.requests if b][1]
    assert "rrrr" not in json.dumps(second)


def test_free_model_exclusions_and_demo_default(fake_server):
    z = {"prompt": "0", "completion": "0"}
    fake_server.models = {"data": [{"id": i, "pricing": z} for i in (
        "thinkingmachines/inkling-1:free", "google/lyria-3:free", "meta/llama-guard-content-safety:free",
        "openrouter/free", "nvidia/nemotron-3.5-super-vl:free", "qwen/qwen3-8b:free")]}
    free = models.openrouter_free_models(force=True)
    assert [m["id"] for m in free] == ["nvidia/nemotron-3.5-super-vl:free", "qwen/qwen3-8b:free"]
    assert models.default_free_model(free) == "openrouter:nvidia/nemotron-3.5-super-vl:free"
