"""Plain-language analysis of saved results, for non-technical viewers (GET /insights).

Every number here is computed from results/*.jsonl and the task file. Text is generated from those
numbers. Nothing is hard-coded, except public model sizes and friendly names.
Only complete runs on the given task set are used. Infra failures (rate limits) are excluded, as in report.py.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from . import report
from .runner import task_set_hash

FRIENDLY = {
    "bedrock:nvidia.nemotron-super-3-120b": ("Nemotron Super", 120),
    "bedrock:qwen.qwen3-32b-v1:0": ("Qwen3 32B", 32),
    "bedrock:google.gemma-3-27b-it": ("Gemma 3 27B", 27),
    "bedrock:openai.gpt-oss-120b-1:0": ("gpt-oss 120B", 120),
    "bedrock:deepseek.v3.2": ("DeepSeek V3.2", 671),
    "bedrock:mistral.mistral-large-3-675b-instruct": ("Mistral Large 3", 675),
}
BASELINE_NAMES = {
    "oracle": "Best possible (every design tried)",
    "6-probe": "Random search (6 random tries)",
    "max-square-ws": "Rule of thumb (biggest square chip)",
    "random": "Random guess",
}


def _label(model: str) -> str:
    if model in FRIENDLY:
        return FRIENDLY[model][0]
    if model in BASELINE_NAMES:
        return BASELINE_NAMES[model]
    return model.split(":", 1)[-1]


def plain_layer(name: str) -> str:
    m = re.fullmatch(r"bert_tiny_(attention_proj|mlp_up|mlp_down)_s(\d+)", name)
    if m:
        part = {"attention_proj": "attention step", "mlp_up": "expand step", "mlp_down": "shrink step"}[m.group(1)]
        return f"Language model {part}, {m.group(2)}-word input"
    return {
        "resnet18_layer4_downsample": "Image model (ResNet-18) downsampling step",
        "mobilenetv2_last_expand": "Phone image model (MobileNetV2) expand step",
        "mobilenetv2_last_project": "Phone image model (MobileNetV2) project step",
    }.get(name, name)


def _design(d: dict | None) -> str:
    if not d:
        return "no answer"
    return f"{d.get('array_h')}×{d.get('array_w')} {d.get('dataflow')}"


def _area(d: dict | None) -> int:
    return int((d or {}).get("array_h", 0) or 0) * int((d or {}).get("array_w", 0) or 0)


HOW_IT_WORKS = [
    {"title": "A real chip-design question",
     "body": "We take one real step from an AI model (a language model or an image model) and ask: what shape of AI "
             "chip runs it fastest? The chip is a grid of tiny calculators, and the grid can't be bigger than a fixed "
             "area budget."},
    {"title": "The AI runs experiments",
     "body": "The AI may test up to 6 designs on SCALE-Sim, a real open-source chip simulator used in research. It "
             "sees how fast each design runs, much as an engineer tests prototypes, then submits its final design."},
    {"title": "Scored against the true best",
     "body": "We simulated every allowed design in advance, so we know the true best one. The AI scores 100% if it "
             "matches it, and less the slower its design is. An over-budget design scores 0."},
]


def build_insights(tasks: list[dict], results_dir) -> dict:
    n_tasks = len(tasks)
    if not n_tasks or not Path(results_dir).exists():
        return {"ready": False, "how_it_works": HOW_IT_WORKS}
    h = task_set_hash(tasks)
    by_id = {t["id"]: t for t in tasks}
    rows = [r for r in report._latest(report.load_results(results_dir))
            if r.get("task_set_hash") == h and not r.get("infra_error") and r.get("task_id") in by_id]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["model"]].append(r)
    groups = {m: rs for m, rs in groups.items() if len(rs) == n_tasks}  # complete runs only: fair comparison

    table = []
    for m, rs in groups.items():
        n = len(rs)
        larger = sum(1 for r in rs if r.get("feasible") and _area(r.get("answer")) > _area(by_id[r["task_id"]]["optimum"]["inputs"]))
        table.append({
            "id": m, "label": _label(m), "baseline": any(r.get("baseline") for r in rs), "n": n,
            "mean_score": sum(r.get("score", 0.0) for r in rs) / n,
            "solve_rate": sum(1 for r in rs if r.get("score", 0.0) >= report.SOLVE_THRESHOLD) / n,
            "feasible_rate": sum(1 for r in rs if r.get("feasible")) / n,
            "mean_sim_calls": sum(r.get("sim_calls_used", 0) for r in rs) / n,
            "oversize_rate": larger / n,
            "no_sim_rate": sum(1 for r in rs if not r.get("sim_calls_used")) / n,
            "params_b": FRIENDLY.get(m, (None, None))[1],
        })
    table.sort(key=lambda x: -x["mean_score"])
    models = [x for x in table if not x["baseline"]]
    base = {x["id"]: x for x in table if x["baseline"]}
    if not models or "6-probe" not in base:
        return {"ready": False, "how_it_works": HOW_IT_WORKS}

    best = models[0]
    probe = base["6-probe"]["mean_score"]
    rand = base["random"]["mean_score"] if "random" in base else None
    beat = [x for x in models if x["mean_score"] > probe]
    headline = {"best_model": best["label"], "best_score": best["mean_score"], "probe_score": probe,
                "random_score": rand, "oracle_score": base.get("oracle", {}).get("mean_score", 1.0),
                "beat_probe": len(beat), "best_vs_probe_pct": (best["mean_score"] / probe - 1) * 100 if probe else None,
                "best_solve_rate": best["solve_rate"]}

    findings = []
    # 1. Experimenting pays.
    exp = [x for x in models if x["mean_sim_calls"] >= 4]
    lazy = [x for x in models if x["mean_sim_calls"] < 2]
    if exp and lazy:
        exp_avg = sum(x["mean_score"] for x in exp) / len(exp)
        worst_lazy = min(lazy, key=lambda x: x["mean_score"])
        findings.append({
            "id": "experiment", "title": "AIs that experiment design better chips",
            "stat": f"{exp_avg:.0%}", "stat_label": "average score of AIs that ran about 5 experiments",
            "body": f"{len(exp)} of {len(models)} AIs ran about 5 tests per task, and all of them beat random search "
                    f"({probe:.0%}). {worst_lazy['label']} skipped testing on {worst_lazy['no_sim_rate']:.0%} of "
                    f"tasks and answered from memory. It scored {worst_lazy['mean_score']:.0%}, worse than six random "
                    f"tries." if all(x["mean_score"] > probe for x in exp) else
                    f"AIs that ran about 5 tests averaged {exp_avg:.0%}; {worst_lazy['label']}, which mostly skipped "
                    f"testing, scored {worst_lazy['mean_score']:.0%}.",
            "takeaway": "When an AI assistant has engineering tools, require it to test before it answers, and check "
                        "that it actually did."})
    # 2. Size is not skill.
    sized = [x for x in models if x["params_b"]]
    if len(sized) >= 2:
        biggest = max(sized, key=lambda x: (x["params_b"], -x["mean_score"]))
        better_smaller = [x for x in sized if x["params_b"] < biggest["params_b"] and x["mean_score"] > biggest["mean_score"]]
        if better_smaller:
            s = max(better_smaller, key=lambda x: x["mean_score"])
            findings.append({
                "id": "size", "title": "A bigger AI is not a better engineer",
                "stat": f"{biggest['params_b'] / s['params_b']:.0f}× smaller",
                "stat_label": f"{s['label']} beat {biggest['label']} (total parameters)",
                "body": f"{s['label']} ({s['params_b']}B parameters) scored {s['mean_score']:.0%}. The largest AI "
                        f"tested, {biggest['label']} ({biggest['params_b']}B), scored {biggest['mean_score']:.0%}. "
                        f"How the AI works the problem mattered more than how big it is.",
                "takeaway": "Choose AI models by testing them on your own task, not by size or price."})
    # 3. "Bigger chip" bias.
    over = sum(x["oversize_rate"] for x in models) / len(models)
    nonsq = sum(1 for t in tasks if t["optimum"]["inputs"]["array_h"] != t["optimum"]["inputs"]["array_w"])
    findings.append({
        "id": "oversize", "title": "AIs over-build the chip",
        "stat": f"{over:.0%}", "stat_label": "of AI designs used more chip area than the best design",
        "body": f"Across all AIs, {over:.0%} of final designs were bigger than they needed to be. More area cost "
                f"more and was often slower. In {nonsq} of {n_tasks} tasks the best chip was a narrow rectangle, "
                f"not a big square.",
        "takeaway": "Check AI design suggestions with a simulator before you build anything. A \"bigger is better\" "
                    "instinct wastes chip area."})
    # 4. Room to improve.
    findings.append({
        "id": "headroom", "title": "Even the best AI often misses the best design",
        "stat": f"{best['solve_rate']:.0%}", "stat_label": f"of tasks where {best['label']} found the best design",
        "body": f"The top AI ({best['label']}) averaged {best['mean_score']:.0%} of the best possible score, but "
                f"found the true best design in only {best['solve_rate']:.0%} of tasks.",
        "takeaway": "This environment scores every attempt automatically, so it can be used to train AIs to "
                    "experiment better (reinforcement learning). That's the next step."})

    scatter = [{"label": x["label"], "mean_sim_calls": x["mean_sim_calls"], "mean_score": x["mean_score"]} for x in models]

    # Hardest workloads: average over models and budgets.
    per_layer: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in (x["id"] for x in models):
        for r in groups[m]:
            per_layer[by_id[r["task_id"]]["layer_name"]][m].append(r.get("score", 0.0))
    workloads = []
    for layer, ms in per_layer.items():
        means = [sum(v) / len(v) for v in ms.values()]
        kind = next(t["layer"]["kind"] for t in tasks if t["layer_name"] == layer)
        workloads.append({"layer": layer, "plain": plain_layer(layer), "kind": kind,
                          "avg_model_score": sum(means) / len(means), "best_model_score": max(means),
                          "worst_model_score": min(means)})
    workloads.sort(key=lambda w: w["avg_model_score"])

    return {"ready": True, "task_set": h, "n_tasks": n_tasks, "n_models": len(models), "headline": headline,
            "rows": table, "findings": findings, "experiment_vs_score": scatter, "workloads": workloads,
            "example": _example(groups, models, by_id), "how_it_works": HOW_IT_WORKS}


def _example(groups, models, by_id) -> dict | None:
    """A real episode in which an AI that did experiment still over-built the chip (the largest area gap wins)."""
    best = None
    for x in models:
        if x["mean_sim_calls"] < 4:
            continue
        for r in groups[x["id"]]:
            opt = by_id[r["task_id"]]["optimum"]["inputs"]
            log = r.get("sim_log") or []
            if not r.get("feasible") or len(log) < 4 or _area(r.get("answer")) <= _area(opt):
                continue
            key = (_area(r["answer"]) / max(_area(opt), 1), -r.get("score", 0), x["mean_score"])
            if best is None or key > best[0]:
                best = (key, x, r)
    if not best:
        return None
    _, x, r = best
    t = by_id[r["task_id"]]
    opt_cycles = t["optimum"]["value"]
    ans_cycles = next((s["cycles"] for s in r["sim_log"] if s["design"] == r["answer"]), None)
    if ans_cycles is None and r.get("score"):
        ans_cycles = round(opt_cycles / r["score"])
    steps = [{"design": _design(s["design"]), "cycles": s["cycles"], "util": s.get("utilization_pct")} for s in r["sim_log"]]
    max_util = max((s["util"] or 0) for s in steps)
    from .packs import accel
    o = accel.simulate(t["layer"], **t["optimum"]["inputs"])
    faster = (1 - opt_cycles / ans_cycles) * 100 if ans_cycles else None
    return {
        "task_plain": plain_layer(t["layer_name"]), "layer": t["layer_name"], "model": x["label"],
        "budget_pe": t["pe_budget"], "steps": steps,
        "answer": {"design": _design(r["answer"]), "cycles": ans_cycles},
        "optimum": {"design": _design(t["optimum"]["inputs"]), "cycles": opt_cycles, "util": o["utilization_pct"]},
        "score": r.get("score", 0.0),
        "lesson": (f"{x['label']} ran {len(steps)} experiments, but its chips were never more than {max_util:.0f}% busy, "
                   f"which means most of the chip sat idle. It still picked {_design(r['answer'])}. The best design was "
                   f"the much smaller {_design(t['optimum']['inputs'])}: {o['utilization_pct']:.0f}% busy and "
                   f"{faster:.0f}% fewer cycles."),
    }
