import math

import numpy as np
import pytest

from sim2gym.packs import accel


@pytest.mark.parametrize("design", [
    None, [], {}, {"array_h": 16},
    dict(array_h=True, array_w=16, dataflow="ws", sram_split="balanced"),
    dict(array_h=16.0, array_w=16, dataflow="ws", sram_split="balanced"),
    dict(array_h=7, array_w=16, dataflow="ws", sram_split="balanced"),
    dict(array_h=64, array_w=64, dataflow="ws", sram_split="balanced"),
    dict(array_h=16, array_w=16, dataflow="bad", sram_split="balanced"),
    dict(array_h=16, array_w=16, dataflow="ws", sram_split="bad"),
    dict(array_h=16, array_w=16, dataflow="ws", sram_split="balanced", extra=1),
])
def test_invalid_designs(design):
    assert accel.validate_design(design, 256) is not None


def test_exhaustive_grid():
    for budget in accel.PE_BUDGETS:
        expected = {(h, w, df, split) for h in accel.ARRAY_DIMS for w in accel.ARRAY_DIMS
                    for df in accel.DATAFLOWS for split in accel.SRAM_SPLITS if h * w <= budget}
        actual = accel.feasible_designs(budget)
        assert {tuple(d.values()) for d in actual} == expected
        assert all(accel.validate_design(d, budget) is None for d in actual)
    assert len({sum(accel.SRAM_KB[s]) for s in accel.SRAM_SPLITS}) == 1
    assert 10 <= len(accel.WORKLOADS) <= 16
    assert all(layer["source"].startswith("https://") for layer in accel.WORKLOADS.values())


@pytest.mark.parametrize("name,df", [("bert_tiny_attention_proj_s8", "os"),
                                     ("bert_tiny_mlp_down_s32", "is")])
def test_cache_equals_real_run(name, df, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(accel, "CACHE_DIR", tmp_path)
    layer = accel.WORKLOADS[name]
    design = dict(array_h=16, array_w=16, dataflow=df, sram_split="balanced")
    fresh = accel._simulate_uncached(layer, **design)
    assert accel.simulate(layer, **design) == fresh
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(accel, "_simulate_uncached", lambda *a, **kw: pytest.fail("Cache miss"))
    assert accel.simulate(layer, **design) == fresh
    assert capsys.readouterr().out == ""


def test_cache_key_covers_inputs_and_version(monkeypatch):
    layer = accel.WORKLOADS["bert_tiny_attention_proj_s8"]
    design = accel.feasible_designs(256)[0]
    before = accel._cache_path(layer, design)
    assert before != accel._cache_path(dict(layer, m=9), design)
    assert before != accel._cache_path(layer, dict(design, dataflow="ws"))
    monkeypatch.setattr(accel, "SCALESIM_VERSION", "different")
    assert before != accel._cache_path(layer, design)


def test_real_simulator_sanity():
    layer = accel.WORKLOADS["bert_tiny_mlp_up_s128"]
    small = accel.simulate(layer, 16, 16, "os", "balanced")
    large = accel.simulate(layer, 32, 32, "os", "balanced")
    assert large["cycles"] <= small["cycles"]
    macs = layer["m"] * layer["n"] * layer["k"]
    for size, result in ((16, small), (32, large)):
        assert result["cycles"] >= math.ceil(macs / size**2)
        assert 0 <= result["stall_cycles"] <= result["cycles"]
        assert 0 < result["utilization_pct"] <= 100
        assert result["utilization_pct"] == pytest.approx(100 * macs / (result["cycles"] * size**2))


def test_numpy_compatibility_keeps_scalar_max_semantics():
    assert accel._scalar_max(np.array([[2], [5], [3]])) == 5
    assert accel._scalar_max(2, 4) == 4
    assert accel._scalar_max([1, 3], default=0) == 3


@pytest.mark.parametrize("layer", [dict(kind="gemm", m=True, n=8, k=8),
                                     dict(kind="gemm", m=0, n=8, k=8), dict(kind="other")])
def test_invalid_workloads(layer):
    with pytest.raises(ValueError):
        accel.simulate(layer, 16, 16, "ws", "balanced")
