"""Procedural AccelGym workloads: sample fresh layer shapes so task sets are unlimited and unseen.

Each seed gives a different, reproducible set of GEMM and convolution layers. Every task still gets an
exhaustive SCALE-Sim optimum, so generated tasks are graded exactly like the fixed ones. Use a new seed for a
held-out eval split that no model can have memorized.
"""

import multiprocessing
import random

from sim2gym.packs import accel
from sim2gym.tasks import build_tasks_for_layers

# Token counts from single-token decode to short prefill; hidden sizes of small transformers.
GEMM_M = [1, 2, 4, 8, 16, 32, 64, 128, 256]
GEMM_KN = [64, 96, 128, 192, 256, 384, 512, 768]
CONV_HW = [7, 14, 28]
CONV_FILTER = [1, 3]
CONV_STRIDE = [1, 2]
CONV_CHANNELS = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]
CONV_SHARE = 0.3
MAX_MACS = 8 * 2 ** 20  # the largest fixed workload (BERT-tiny MLP at sequence 128); keeps SCALE-Sim fast
SOURCE = "sim2gym procedural generator, seed {seed}"


def layer_macs(layer: dict) -> int:
    if layer["kind"] == "gemm":
        return layer["m"] * layer["k"] * layer["n"]
    out_h = (layer["ifmap_h"] - layer["filter_h"]) // layer["stride"] + 1
    out_w = (layer["ifmap_w"] - layer["filter_w"]) // layer["stride"] + 1
    return out_h * out_w * layer["filters"] * layer["channels"] * layer["filter_h"] * layer["filter_w"]


def _shape(layer: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in layer.items() if k not in ("name", "source")))


def _sample(rng: random.Random) -> dict:
    if rng.random() < CONV_SHARE:
        hw, f = rng.choice(CONV_HW), rng.choice(CONV_FILTER)
        return dict(kind="conv", ifmap_h=hw, ifmap_w=hw, filter_h=f, filter_w=f,
                    channels=rng.choice(CONV_CHANNELS), filters=rng.choice(CONV_CHANNELS),
                    stride=rng.choice(CONV_STRIDE))
    return dict(kind="gemm", m=rng.choice(GEMM_M), k=rng.choice(GEMM_KN), n=rng.choice(GEMM_KN))


def _name(seed: int, i: int, layer: dict) -> str:
    if layer["kind"] == "gemm":
        return f"gen{seed}_{i:03d}_gemm_m{layer['m']}_k{layer['k']}_n{layer['n']}"
    return (f"gen{seed}_{i:03d}_conv_{layer['ifmap_h']}x{layer['ifmap_w']}_f{layer['filter_h']}"
            f"_c{layer['channels']}_o{layer['filters']}_s{layer['stride']}")


def sample_layers(seed: int, n: int) -> list[dict]:
    """``n`` distinct layers for ``seed``, never repeating a fixed workload's shape."""
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    rng = random.Random(f"sim2gym-generate:{seed}")
    seen = {_shape(layer) for layer in accel.WORKLOADS.values()}
    layers = []
    for _ in range(1000 * n):
        layer = _sample(rng)
        if layer_macs(layer) > MAX_MACS or _shape(layer) in seen:
            continue
        seen.add(_shape(layer))
        layers.append(dict(layer, name=_name(seed, len(layers), layer), source=SOURCE.format(seed=seed)))
        if len(layers) == n:
            return layers
    raise RuntimeError(f"could not sample {n} distinct layers")


def precompute(layers: list[dict], workers: int = 4) -> int:
    """Run SCALE-Sim once per layer and design, in parallel, filling the shared cache."""
    jobs = [(layer, design) for layer in layers for design in accel.feasible_designs(max(accel.PE_BUDGETS))]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(accel._precompute_one, jobs):
            pass
    return len(jobs)


def generate_tasks(seed: int, n: int, budget: int = 6, pe_budgets: list[int] | None = None,
                   workers: int = 1) -> list[dict]:
    layers = sample_layers(seed, n)
    if workers > 1:
        precompute(layers, workers)
    return build_tasks_for_layers(layers, seed=seed, budget=budget, pe_budgets=pe_budgets)
