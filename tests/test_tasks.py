import json

import pytest

from sim2gym.packs import accel
from sim2gym.tasks import build_tasks, load_tasks, public_task, save_tasks


def test_reproducible_task_files(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    tasks = build_tasks(seed=0)
    save_tasks(first, tasks)
    save_tasks(second, build_tasks(seed=0))
    assert first.read_bytes() == second.read_bytes()
    assert load_tasks(first) == tasks
    assert len(tasks) == len(accel.WORKLOADS) * len(accel.PE_BUDGETS)
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert {t["id"] for t in build_tasks(seed=1)} == {t["id"] for t in tasks}
    assert build_tasks(seed=1) != tasks
    assert all(t["sim_version"] == accel.SIM_VERSION for t in tasks)


def test_optimum_is_exhaustive():
    for task in build_tasks():
        candidates = [(accel.simulate(task["layer"], **d)["cycles"], d)
                      for d in accel.feasible_designs(task["pe_budget"])]
        cycles, design = min(candidates, key=lambda p: (p[0], p[1]["array_h"] * p[1]["array_w"],
                            p[1]["array_h"], p[1]["array_w"], p[1]["dataflow"], p[1]["sram_split"]))
        assert task["optimum"] == dict(inputs=design, value=cycles, method="exhaustive")


def test_public_task_has_no_answer():
    task = build_tasks(workloads=[next(iter(accel.WORKLOADS))], pe_budgets=[256])[0]
    task["expected"] = "secret"
    public = public_task(task)
    assert "optimum" not in public and "expected" not in public
    public["space"]["array_h"].clear()
    assert task["space"]["array_h"] == accel.ARRAY_DIMS
    assert "optimum" not in task["prompt"].lower()
    assert "last line" in task["prompt"]
    for field in task["free"]:
        assert field in task["prompt"]
    json.dumps(public)


def test_task_filters_and_empty_selection():
    name = next(iter(accel.WORKLOADS))
    tasks = build_tasks(budget=3, workloads=[name], pe_budgets=[256])
    assert len(tasks) == 1 and tasks[0]["budget"] == 3
    assert build_tasks(workloads=[]) == []
    assert build_tasks(pe_budgets=[]) == []
    with pytest.raises(ValueError):
        build_tasks(budget=-1)
    with pytest.raises(ValueError):
        build_tasks(pe_budgets=[123])
