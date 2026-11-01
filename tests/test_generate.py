"""Procedural workloads: reproducible, distinct, valid, never a fixed workload, graded exactly."""
import pytest

from sim2gym import generate, tasks
from sim2gym.grading import grade_design
from sim2gym.packs import accel

from test_env import fake_simulate


def test_sample_is_reproducible_and_seed_dependent():
    assert generate.sample_layers(1, 20) == generate.sample_layers(1, 20)
    assert generate.sample_layers(1, 20) != generate.sample_layers(2, 20)
    assert generate.sample_layers(1, 5) == generate.sample_layers(1, 20)[:5]


def test_layers_are_valid_distinct_and_new():
    layers = generate.sample_layers(7, 60)
    fixed = {generate._shape(l) for l in accel.WORKLOADS.values()}
    shapes = [generate._shape(l) for l in layers]
    assert len(set(shapes)) == 60 and not fixed & set(shapes)
    assert len({l["name"] for l in layers}) == 60
    assert {l["kind"] for l in layers} == {"gemm", "conv"}
    for layer in layers:
        accel._topology_values(layer)  # raises on an invalid SCALE-Sim layer
        assert generate.layer_macs(layer) <= generate.MAX_MACS
        assert set(layer) - {"name", "source", "kind"} == (
            {"m", "k", "n"} if layer["kind"] == "gemm" else
            {"ifmap_h", "ifmap_w", "filter_h", "filter_w", "channels", "filters", "stride"})


def test_layer_macs_matches_scalesim_conv_output():
    conv = dict(kind="conv", ifmap_h=14, ifmap_w=14, filter_h=3, filter_w=3, channels=16, filters=32, stride=2)
    assert generate.layer_macs(conv) == 6 * 6 * 32 * 16 * 9
    assert generate.layer_macs(dict(kind="gemm", m=4, k=64, n=96)) == 4 * 64 * 96


def test_generated_tasks_have_exact_optima(monkeypatch):
    monkeypatch.setattr(accel, "simulate", fake_simulate)
    ts = generate.generate_tasks(3, 4)
    assert len(ts) == 4 * len(accel.PE_BUDGETS) and len({t["id"] for t in ts}) == len(ts)
    for t in ts:
        assert t["layer_name"].startswith("gen3_") and t["optimum"]["method"] == "exhaustive"
        assert grade_design(t, t["optimum"]["inputs"])["score"] == 1.0
        best = min(fake_simulate(t["layer"], **d)["cycles"] for d in accel.feasible_designs(t["pe_budget"]))
        assert t["optimum"]["value"] == best
        assert "optimum" not in tasks.public_task(t) and "seed 3" in t["prompt"]


def test_fixed_task_set_unchanged_by_refactor():
    fixed = tasks.load_tasks("tasks.jsonl")
    assert sorted(tasks.build_tasks(seed=0), key=lambda t: t["id"]) == sorted(fixed, key=lambda t: t["id"])


@pytest.mark.parametrize("n", [0, -1, 1.5])
def test_rejects_bad_count(n):
    with pytest.raises(ValueError):
        generate.sample_layers(0, n)
