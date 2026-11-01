"""CLI smoke: build -> run oracle/random/6-probe -> report, on real SCALE-Sim (disk-cached) for one small layer."""
import json

from sim2gym import cli

WL = "bert_tiny_attention_proj_s8"


def test_build_run_report(tmp_path, capsys):
    tp, res, lb = tmp_path / "tasks.jsonl", tmp_path / "results", tmp_path / "leaderboard.md"
    assert cli.main(["build", "--seed", "0", "--workloads", WL, "--pe-budgets", "256", "-o", str(tp)]) == 0
    t = json.loads(tp.read_text().splitlines()[0])
    assert t["optimum"]["method"] == "exhaustive" and t["budget"] == 6
    for m in ("oracle", "random", "6-probe"):
        assert cli.main(["run", str(tp), "--model", m, "--results", str(res)]) == 0
    oracle = json.loads((res / "oracle.jsonl").read_text().splitlines()[0])
    assert oracle["score"] == 1.0 and oracle["answer_cycles"] == t["optimum"]["value"]
    capsys.readouterr()
    assert cli.main(["report", str(res), "-o", str(lb), "--tasks", str(tp)]) == 0
    md = lb.read_text()
    assert "| oracle | baseline | 1 | 1.000 | 100% | 100% |" in md and "6-probe" in md and "random" in md
    assert "WARNING" not in md


def test_report_warns_on_mixed_task_sets(tmp_path, capsys):
    res = tmp_path / "results"
    res.mkdir()
    base = {"model": "oracle", "baseline": True, "score": 1.0, "feasible": True, "sim_calls_used": 0, "latency_s": 0}
    (res / "a.jsonl").write_text(json.dumps(dict(base, task_id="t1", task_set_hash="aaa")) + "\n" +
                                 json.dumps(dict(base, task_id="t1", task_set_hash="bbb")) + "\n")
    assert cli.main(["report", str(res), "-o", str(tmp_path / "lb.md")]) == 0
    assert "different task sets" in (tmp_path / "lb.md").read_text()


def test_list(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert WL in out and "6-probe" in out
