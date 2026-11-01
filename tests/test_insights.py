from pathlib import Path

import pytest

from sim2gym import report
from sim2gym.insights import build_insights, plain_layer
from sim2gym.runner import task_set_hash
from sim2gym.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_not_ready_without_results(tmp_path):
    assert build_insights([], tmp_path)["ready"] is False
    assert build_insights(load_tasks(ROOT / "tasks.jsonl"), tmp_path / "missing")["ready"] is False


def test_plain_names():
    assert plain_layer("bert_tiny_mlp_down_s8") == "Language model shrink step, 8-word input"
    assert "ResNet-18" in plain_layer("resnet18_layer4_downsample")


@pytest.mark.skipif(not (ROOT / "results").exists(), reason="no saved results")
def test_insights_numbers_match_leaderboard():
    tasks = load_tasks(ROOT / "tasks.jsonl")
    d = build_insights(tasks, ROOT / "results")
    assert d["ready"] and d["n_tasks"] == len(tasks)
    lb = {r["model"]: r for r in report.build_leaderboard(ROOT / "results", task_set_hash(tasks))["rows"]}
    for row in d["rows"]:  # every number traces back to the same saved results as the leaderboard
        assert row["n"] == len(tasks)
        assert row["mean_score"] == pytest.approx(lb[row["id"]]["mean_score"])
    best = next(r for r in d["rows"] if not r["baseline"])
    assert d["headline"]["best_score"] == pytest.approx(best["mean_score"])
    assert d["headline"]["probe_score"] == pytest.approx(lb["6-probe"]["mean_score"])
    assert {f["id"] for f in d["findings"]} >= {"oversize", "headroom"}
    ex = d["example"]
    assert ex is None or (ex["answer"]["cycles"] >= ex["optimum"]["cycles"] and len(ex["steps"]) >= 4)
