import itertools
import json

from sim2gym import report
from sim2gym.baselines import expected_probe
from sim2gym.grading import grade_design
from sim2gym.packs import accel
from sim2gym.tasks import build_tasks


def _task():
    return build_tasks(workloads=["resnet18_layer4_downsample"], pe_budgets=[256])[0]


def test_expected_probe_matches_brute_force():
    task = _task()
    scores = [grade_design(task, d)["score"] for d in accel.feasible_designs(task["pe_budget"])]
    for k in (1, 2, 6):
        draws = [max(c) for c in itertools.combinations(scores, k)]
        exact = expected_probe(task, k)
        assert abs(exact["mean_score"] - sum(draws) / len(draws)) < 1e-12
        assert abs(exact["solve_rate"] - sum(d >= 0.95 for d in draws) / len(draws)) < 1e-12


def test_expected_probe_bounds():
    task = _task()
    assert expected_probe(task, 1)["mean_score"] < expected_probe(task, 6)["mean_score"] <= 1.0
    everything = expected_probe(task, 10_000)
    assert everything["mean_score"] == 1.0 and everything["solve_rate"] == 1.0
    assert everything["probes"] == len(accel.feasible_designs(task["pe_budget"]))


def test_report_adds_exact_rows_and_ci(tmp_path):
    task = _task()
    res = tmp_path / "results"
    res.mkdir()
    rows = [dict(model="m", task_set_hash="h", task_id=task["id"], score=s, feasible=True) for s in (0.5, 1.0)]
    rows[1]["task_id"] = "other"
    (res / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    lb = report.build_leaderboard(res, "h", [task])
    by = {r["model"]: r for r in lb["rows"]}
    assert by["6-probe (exact expectation)"]["mean_score"] == expected_probe(task, 6)["mean_score"]
    assert by["random (exact expectation)"]["mean_score"] == expected_probe(task, 1)["mean_score"]
    assert abs(by["m"]["ci95"] - 1.96 * (0.125 / 2) ** 0.5) < 1e-12  # sample variance of [0.5, 1.0] is 0.125
    md = report.to_markdown(lb)
    assert "| 95% CI |" in md and "| exact |" in md
