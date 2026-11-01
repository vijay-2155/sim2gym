import pytest

from sim2gym.grading import grade_design
from sim2gym.packs import accel
from sim2gym.tasks import build_tasks


def test_oracle_scores_one_on_every_task():
    for task in build_tasks():
        result = grade_design(task, task["optimum"]["inputs"])
        assert result == dict(score=1.0, solved=True, feasible=True,
                              cycles=task["optimum"]["value"], error=None)


@pytest.mark.parametrize("design", [None, "wrong", 17, [], {},
    dict(array_h=64, array_w=64, dataflow="ws", sram_split="balanced"),
    dict(array_h=16, array_w=16, dataflow="wrong", sram_split="balanced"),
    dict(array_h=False, array_w=16, dataflow="ws", sram_split="balanced"),
    dict(array_h=16, array_w=16, dataflow="ws", sram_split="balanced", cycles=1),
])
def test_invalid_and_over_budget_score_zero(design):
    task = build_tasks(workloads=[next(iter(accel.WORKLOADS))], pe_budgets=[256])[0]
    result = grade_design(task, design)
    assert result["score"] == 0 and not result["feasible"] and not result["solved"]
    assert result["cycles"] is None and result["error"]


def test_valid_suboptimal_design_gets_ratio():
    task = build_tasks(workloads=["bert_tiny_mlp_up_s128"], pe_budgets=[1024])[0]
    design = accel.feasible_designs(1024)[0]
    result = grade_design(task, design)
    assert 0 < result["score"] < 1
    assert result["score"] == task["optimum"]["value"] / result["cycles"]
    assert result["solved"] == (result["score"] >= 0.95)
