"""kiro:<model> and codex:<model> adapters, driven by fake CLI scripts (no network, no accounts)."""
import json
import os
import stat
from pathlib import Path

from sim2gym import models, runner, tasks

REPO = Path(__file__).resolve().parents[1]

FAKE = """#!/usr/bin/env python3
import json, os, sys
log = os.environ["FAKE_KIRO_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "files": os.listdir("."),
                        "no_color": os.environ.get("NO_COLOR")}) + "\\n")
mode = os.environ.get("FAKE_KIRO_MODE", "ok")
if mode == "fail":
    print("boom", file=sys.stderr); sys.exit(3)
print("\\x1b[32m" + '{"answer": {"array_h": 16, "array_w": 16, "dataflow": "ws", "sram_split": "balanced"}}' + "\\x1b[0m")
"""


def _fake(tmp_path, monkeypatch, mode="ok"):
    exe = tmp_path / "kiro-cli"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_KIRO_LOG", str(log))
    monkeypatch.setenv("FAKE_KIRO_MODE", mode)
    return str(exe), log


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_get_model_kiro_and_paid():
    m = models.get_model("kiro:claude-opus-5.5", max_tokens=100, timeout_s=9)
    assert isinstance(m, models.KiroModel) and m.name == "kiro:claude-opus-5.5" and m.timeout_s == 9
    assert runner.is_paid("kiro:claude-opus-5.5") and runner.default_workers("kiro:x") == 4


def test_render_transcript_first_turn_is_prompt_unchanged():
    assert models.render_transcript([{"role": "user", "content": "TASK"}]) == "TASK"
    later = models.render_transcript([{"role": "user", "content": "TASK"}, {"role": "assistant", "content": "A1"},
                                      {"role": "user", "content": "OBS"}])
    assert later.index("TASK") < later.index("[ASSISTANT]\nA1") < later.index("[USER]\nOBS")
    assert later.rstrip().endswith("Do not repeat the label.")


def test_chat_runs_in_empty_dir_with_no_tools(tmp_path, monkeypatch):
    exe, log = _fake(tmp_path, monkeypatch)
    r = models.KiroModel("claude-opus-5.5", binary=exe, retries=0).chat([{"role": "user", "content": "TASK"}])
    assert r.error is None and r.text.startswith('{"answer"') and "\x1b" not in r.text
    (call,) = _calls(log)
    assert call["argv"] == ["chat", "--no-interactive", "--model", "claude-opus-5.5", "--trust-tools=", "TASK"]
    assert call["files"] == [] and os.path.realpath(call["cwd"]) != os.path.realpath(os.getcwd())
    assert call["no_color"] == "1" and not os.path.exists(call["cwd"])  # temp dir removed afterwards


def test_chat_failure_retries_then_errors(tmp_path, monkeypatch):
    exe, log = _fake(tmp_path, monkeypatch, mode="fail")
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    r = models.KiroModel("m", binary=exe, retries=1).chat([{"role": "user", "content": "TASK"}])
    assert r.text == "" and r.error.startswith("kiro error: exit 3") and "boom" in r.error
    assert len(_calls(log)) == 2


def test_missing_binary_is_an_error():
    r = models.KiroModel("m", binary="/nonexistent/kiro-cli").chat([{"role": "user", "content": "x"}])
    assert r.error == "kiro error: /nonexistent/kiro-cli not found"


def test_episode_end_to_end_with_fake_kiro(tmp_path, monkeypatch):
    task = tasks.load_tasks(REPO / "tasks.jsonl")[0]  # graded from the checked-in SCALE-Sim cache
    exe, _ = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("SIM2GYM_KIRO_BIN", exe)
    pol = runner.LLMPolicy(models.get_model("kiro:m"))
    r = runner.run_episode(task, pol, "kiro:m")
    assert r["feasible"] and r["answer"]["array_h"] == 16 and 0 < r["score"] <= 1.0


FAKE_CODEX = """#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["FAKE_KIRO_LOG"], "a") as f:
    f.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "files": os.listdir(".")}) + "\\n")
print("codex banner and token counts go to stdout; they must be ignored")
with open(argv[argv.index("-o") + 1], "w") as f:
    f.write('{"answer": {"array_h": 16, "array_w": 16, "dataflow": "ws", "sram_split": "balanced"}}')
"""


def test_codex_reads_last_message_with_tools_disabled(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(FAKE_CODEX)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_KIRO_LOG", str(log))
    m = models.get_model("codex:gpt-6.1-sol")
    assert isinstance(m, models.CodexModel) and m.name == "codex:gpt-6.1-sol" and runner.is_paid(m.name)
    m.binary = str(exe)
    r = m.chat([{"role": "user", "content": "TASK"}])
    assert r.error is None and r.text.startswith('{"answer"') and "banner" not in r.text
    (call,) = _calls(log)
    argv = call["argv"]
    assert argv[:2] == ["exec", "--ephemeral"] and argv[-2:] == ["--", "TASK"]
    assert "--ignore-user-config" in argv and argv[argv.index("-s") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "gpt-6.1-sol" and argv[argv.index("-C") + 1] == call["cwd"]
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    assert disabled == set(models.CODEX_DISABLED_FEATURES) and "shell_tool" in disabled
    assert call["files"] == [] and not os.path.exists(call["cwd"])
