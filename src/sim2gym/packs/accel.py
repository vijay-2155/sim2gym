"""A small, exhaustive accelerator design space backed by SCALE-Sim 3.

Cycles and utilization use COMPUTE_REPORT's Total Cycles / Overall Util %
(compute plus stalls; initial prefetch and final DRAM drain are excluded).
"""

import contextlib
import builtins
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

SCALESIM_VERSION = importlib.metadata.version("scalesim")
SIM_NAME = "accel"
ARRAY_DIMS = [8, 16, 32, 64]
DATAFLOWS = ["os", "ws", "is"]
SRAM_SPLITS = ["balanced"]
PE_BUDGETS = [256, 1024, 4096]
SRAM_KB = {"balanced": (64, 64, 64)}
BANDWIDTH = 32  # Words/cycle, fixed for all three memory interfaces.
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"

_BERT_SOURCE = "https://huggingface.co/prajjwal1/bert-tiny/blob/main/config.json"
_RESNET_SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"
_MOBILE_SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py"
WORKLOADS = {}
for _seq in (8, 32, 128):
    for _op, _k, _n in (("attention_proj", 128, 128), ("mlp_up", 128, 512), ("mlp_down", 512, 128)):
        _name = f"bert_tiny_{_op}_s{_seq}"
        WORKLOADS[_name] = dict(name=_name, kind="gemm", source=_BERT_SOURCE,
                                m=_seq, n=_n, k=_k)
WORKLOADS.update({
    "resnet18_layer4_downsample": dict(
        name="resnet18_layer4_downsample", kind="conv", source=_RESNET_SOURCE,
        ifmap_h=14, ifmap_w=14, filter_h=1, filter_w=1, channels=256, filters=512, stride=2),
    "mobilenetv2_last_expand": dict(
        name="mobilenetv2_last_expand", kind="conv", source=_MOBILE_SOURCE,
        ifmap_h=7, ifmap_w=7, filter_h=1, filter_w=1, channels=160, filters=960, stride=1),
    "mobilenetv2_last_project": dict(
        name="mobilenetv2_last_project", kind="conv", source=_MOBILE_SOURCE,
        ifmap_h=7, ifmap_w=7, filter_h=1, filter_w=1, channels=960, filters=320, stride=1),
})


def describe_layer(layer: dict) -> str:
    if layer["kind"] == "gemm":
        shape = f"GEMM: ({layer['m']} x {layer['k']}) times ({layer['k']} x {layer['n']})"
    else:
        shape = (f"convolution: input {layer['ifmap_h']} x {layer['ifmap_w']} x {layer['channels']}, "
                 f"filter {layer['filter_h']} x {layer['filter_w']}, "
                 f"{layer['filters']} output channels, stride {layer['stride']}, no padding")
    return f"{layer['name']}: {shape}. Source: {layer['source']}"


def validate_design(design: dict, pe_budget: int) -> str | None:
    if not isinstance(design, dict) or set(design) != {"array_h", "array_w", "dataflow", "sram_split"}:
        return "Expected exactly array_h, array_w, dataflow, sram_split"
    for key in ("array_h", "array_w"):
        if type(design[key]) is not int or design[key] not in ARRAY_DIMS:
            return f"{key} must be an allowed integer"
    if design["dataflow"] not in DATAFLOWS or design["sram_split"] not in SRAM_SPLITS:
        return "Unknown dataflow or SRAM split"
    if design["array_h"] * design["array_w"] > pe_budget:
        return "PE budget exceeded"
    return None


def feasible_designs(pe_budget: int) -> list[dict]:
    return [dict(array_h=h, array_w=w, dataflow=df, sram_split=split)
            for h in ARRAY_DIMS for w in ARRAY_DIMS if h * w <= pe_budget
            for df in DATAFLOWS for split in SRAM_SPLITS]


def _topology_values(layer):
    if layer.get("kind") == "gemm":
        values = [layer[key] for key in ("m", "n", "k")]
    elif layer.get("kind") == "conv":
        values = [layer[key] for key in
                  ("ifmap_h", "ifmap_w", "filter_h", "filter_w", "channels", "filters", "stride")]
        if values[2] > values[0] or values[3] > values[1]:
            raise ValueError("Filter exceeds input dimensions")
    else:
        raise ValueError("Expected a gemm or conv layer")
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("Layer dimensions must be positive integers")
    return values


def _simulate_uncached(layer, array_h, array_w, dataflow, sram_split):
    from scalesim.scale_sim import scalesim
    from scalesim.memory import read_buffer, double_buffered_scratchpad_mem

    # SCALE-Sim 3 casts singleton arrays to int, disallowed by NumPy >= 2.4.
    read_buffer.max = _scalar_max
    double_buffered_scratchpad_mem.max = _scalar_max

    values = _topology_values(layer)
    ifmap, filt, ofmap = SRAM_KB[sram_split]
    with tempfile.TemporaryDirectory(prefix="sim2gym-") as directory:
        root = Path(directory)
        cfg = root / "array.cfg"
        cfg.write_text(f"""[general]
run_name = accel
[run_presets]
InterfaceBandwidth = USER
UseRamulatorTrace = False
[architecture_presets]
ArrayHeight = {array_h}
ArrayWidth = {array_w}
IfmapSramSzkB = {ifmap}
FilterSramSzkB = {filt}
OfmapSramSzkB = {ofmap}
IfmapOffset = 0
FilterOffset = 10000000
OfmapOffset = 20000000
Dataflow = {dataflow}
Bandwidth = {BANDWIDTH}
ReadRequestBuffer = 64
WriteRequestBuffer = 64
[layout]
IfmapCustomLayout = False
FilterCustomLayout = False
IfmapSRAMBankBandwidth = 10
IfmapSRAMBankNum = 10
IfmapSRAMBankPort = 2
FilterSRAMBankBandwidth = 10
FilterSRAMBankNum = 10
FilterSRAMBankPort = 2
[sparsity]
SparsitySupport = False
""", encoding="utf-8")
        topology = root / "topology.csv"
        # SCALE-Sim discards the final CSV field, so retain the trailing comma.
        topology.write_text("layer,dimensions,\nlayer," + ",".join(map(str, values)) + ",\n", encoding="utf-8")
        layout = root / "layout.csv"
        layout.write_text("layer,\n", encoding="utf-8")
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            sim = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                           topology=str(topology), layout=str(layout), input_type_gemm=layer["kind"] == "gemm")
            sim.run_scale(top_path=str(root / "out"))
        with (root / "out" / "accel" / "COMPUTE_REPORT.csv").open() as report:
            row = next(csv.DictReader(report, skipinitialspace=True))
        result = dict(cycles=int(float(row["Total Cycles"])),
                      stall_cycles=int(float(row["Stall Cycles"])),
                      utilization_pct=float(row["Overall Util %"]))
        if (result["cycles"] <= 0 or not 0 <= result["stall_cycles"] <= result["cycles"]
                or not math.isfinite(result["utilization_pct"]) or not 0 <= result["utilization_pct"] <= 100):
            raise RuntimeError(f"Invalid SCALE-Sim report: {result}")
        return result


def _scalar_max(*args, **kwargs):
    import numpy as np

    value = builtins.max(*args, **kwargs)
    return value.item() if isinstance(value, np.ndarray) and value.size == 1 else value


def _cache_path(layer, design):
    payload = dict(layer=layer, design=design, scalesim=SCALESIM_VERSION,
                   wrapper=SIM_VERSION, sram=SRAM_KB, bandwidth=BANDWIDTH)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def simulate(layer: dict, array_h: int, array_w: int, dataflow: str, sram_split: str) -> dict:
    design = dict(array_h=array_h, array_w=array_w, dataflow=dataflow, sram_split=sram_split)
    error = validate_design(design, max(PE_BUDGETS))
    if error:
        raise ValueError(error)
    _topology_values(layer)
    path = _cache_path(layer, design)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = _simulate_uncached(layer, **design)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also permits independent evaluators to share the cache.
    with tempfile.NamedTemporaryFile(mode="w", dir=CACHE_DIR, suffix=".tmp", delete=False) as output:
        temp_path = Path(output.name)
        try:
            json.dump(result, output, sort_keys=True, allow_nan=False)
            output.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return result


def _precompute_one(args):
    layer, design = args
    simulate(layer, **design)


def precompute(workers: int = 6):
    """Populate every feasible workload/design pair once, across all PE budgets."""
    started = time.perf_counter()
    jobs = [(layer, design) for layer in WORKLOADS.values()
            for design in feasible_designs(max(PE_BUDGETS))]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(_precompute_one, jobs):
            pass
    return dict(simulations=len(jobs), workers=workers, elapsed_s=time.perf_counter() - started)


SIM_VERSION = SCALESIM_VERSION + "+" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    print(json.dumps(precompute()))
