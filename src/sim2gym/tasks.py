"""Deterministic design tasks with exhaustive SCALE-Sim optima."""

import copy
import json
from pathlib import Path
import random

from sim2gym.packs import accel
from sim2gym.protocol import render_prompt


def build_tasks(seed: int = 0, budget: int = 6, workloads: list[str] | None = None,
                pe_budgets: list[int] | None = None) -> list[dict]:
    if type(budget) is not int or budget < 0:
        raise ValueError("Simulator call budget must be a nonnegative integer")
    names = sorted(accel.WORKLOADS if workloads is None else workloads)
    budgets = sorted(accel.PE_BUDGETS if pe_budgets is None else pe_budgets)
    if any(type(pe) is not int or pe not in accel.PE_BUDGETS for pe in budgets):
        raise ValueError("Unknown PE budget")
    tasks = []
    for name in names:
        layer = accel.WORKLOADS[name]
        results = [(design, accel.simulate(layer, **design)["cycles"])
                   for design in accel.feasible_designs(max(budgets))] if budgets else []
        for pe in budgets:
            design, cycles = min((pair for pair in results
                                  if pair[0]["array_h"] * pair[0]["array_w"] <= pe),
                                 key=lambda pair: (pair[1], pair[0]["array_h"] * pair[0]["array_w"],
                                                   pair[0]["array_h"], pair[0]["array_w"],
                                                   pair[0]["dataflow"], pair[0]["sram_split"]))
            task = dict(id=f"accel/design/{name}/pe{pe}", sim=accel.SIM_NAME,
                        sim_version=accel.SIM_VERSION, kind="design", seed=seed,
                        layer_name=name, layer=copy.deepcopy(layer), pe_budget=pe,
                        space=dict(array_h=list(accel.ARRAY_DIMS), array_w=list(accel.ARRAY_DIMS),
                                   dataflow=list(accel.DATAFLOWS), sram_split=list(accel.SRAM_SPLITS)),
                        free=["array_h", "array_w", "dataflow", "sram_split"],
                        objective={"minimize": "cycles"},
                        optimum=dict(inputs=design.copy(), value=cycles, method="exhaustive"), budget=budget)
            task["prompt"] = render_prompt(task)
            tasks.append(task)
    random.Random(seed).shuffle(tasks)
    return tasks


def save_tasks(path, tasks) -> None:
    with Path(path).open("w", encoding="utf-8") as output:
        for task in tasks:
            output.write(json.dumps(task, sort_keys=True, allow_nan=False) + "\n")


def load_tasks(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def public_task(task: dict) -> dict:
    fields = ("id", "sim", "sim_version", "kind", "seed", "layer_name", "layer", "pe_budget",
              "space", "free", "objective", "budget", "prompt")
    return copy.deepcopy({key: task[key] for key in fields if key in task})
