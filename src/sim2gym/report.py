"""Leaderboard from results/*.jsonl.

Rows are per (model, task_set_hash). Only rows on the same task set are comparable: if more than
one task-set hash is present the report warns and splits the table per task set, unless a
``task_set_hash`` filter is given (e.g. the hash of the tasks file being reported on).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SOLVE_THRESHOLD = 0.95


def load_results(results_dir) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest(rows: list[dict]) -> list[dict]:
    """Keep the last line per (model, task_set_hash, task_id), so resumed/retried runs don't double count."""
    by: dict[tuple, dict] = {}
    for r in rows:
        by[(r.get("model"), r.get("task_set_hash"), r.get("task_id"))] = r
    return list(by.values())


def _ci95(scores: list[float]) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean score."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    var = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    return 1.96 * math.sqrt(var / len(scores))


def expected_baseline_rows(tasks: list[dict], task_set_hash: str | None) -> list[dict]:
    """Exact expectations of the random baselines over every possible draw (no seed luck)."""
    from sim2gym.baselines import expected_probe

    rows = []
    for name, probes in (("random (exact expectation)", 1), ("6-probe (exact expectation)", 6)):
        per_task = [expected_probe(t, min(probes, int(t.get("budget", probes)))) for t in tasks]
        n = len(per_task)
        rows.append({
            "model": name, "task_set_hash": task_set_hash, "baseline": True, "n": n,
            "mean_score": sum(e["mean_score"] for e in per_task) / n, "ci95": 0.0,
            "solve_rate": sum(e["solve_rate"] for e in per_task) / n, "feasible_rate": 1.0,
            "mean_sim_calls": 0.0 if probes == 1 else sum(e["probes"] for e in per_task) / n,
            "mean_latency_s": 0.0, "errors": 0,
        })
    return rows


def build_leaderboard(results_dir, task_set_hash: str | None = None, tasks: list[dict] | None = None) -> dict:
    rows = _latest(load_results(results_dir))
    warnings: list[str] = []
    if task_set_hash:
        dropped = sum(1 for r in rows if r.get("task_set_hash") != task_set_hash)
        rows = [r for r in rows if r.get("task_set_hash") == task_set_hash]
        if dropped:
            warnings.append(f"ignored {dropped} results from other task sets")
    hashes = Counter(r.get("task_set_hash") for r in rows)
    if len(hashes) > 1:
        warnings.append("results come from %d different task sets (%s); scores are only comparable within a "
                        "task set" % (len(hashes), ", ".join(f"{h}: {n}" for h, n in hashes.most_common())))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("model"), r.get("task_set_hash"))].append(r)
    table = []
    for (model, h), all_rs in groups.items():
        # Infra failures (rate limits, provider errors) are not the model's answer: exclude them from scores
        # and report them as incomplete instead of counting them as 0.
        errs = sum(1 for r in all_rs if r.get("infra_error"))
        rs = [r for r in all_rs if not r.get("infra_error")]
        n = len(rs)
        if n == 0:
            continue
        table.append({
            "model": model, "task_set_hash": h, "baseline": any(r.get("baseline") for r in rs), "n": n,
            "mean_score": sum(r.get("score", 0.0) for r in rs) / n,
            "ci95": _ci95([r.get("score", 0.0) for r in rs]),
            "solve_rate": sum(1 for r in rs if r.get("score", 0.0) >= SOLVE_THRESHOLD) / n,
            "feasible_rate": sum(1 for r in rs if r.get("feasible")) / n,
            "mean_sim_calls": sum(r.get("sim_calls_used", 0) for r in rs) / n,
            "mean_latency_s": sum(r.get("latency_s", 0.0) for r in rs) / n,
            "errors": errs,
        })
    if tasks:
        from sim2gym.runner import task_set_hash as hash_tasks
        table += expected_baseline_rows(tasks, task_set_hash or hash_tasks(tasks))
    table.sort(key=lambda x: (x["task_set_hash"] or "", -x["mean_score"], x["model"] or ""))
    ns = {x["task_set_hash"]: set() for x in table}
    for x in table:
        ns[x["task_set_hash"]].add(x["n"])
    for h, s in ns.items():
        if len(s) > 1:
            warnings.append(f"task set {h}: models were run on different numbers of tasks {sorted(s)}")
    return {"rows": table, "warnings": warnings, "task_set_hashes": list(hashes)}


def to_markdown(lb: dict) -> str:
    out = ["# sim2gym leaderboard (AccelGym on SCALE-Sim)", ""]
    for w in lb["warnings"]:
        out.append(f"> WARNING: {w}")
    if lb["warnings"]:
        out.append("")
    by_hash: dict = defaultdict(list)
    for r in lb["rows"]:
        by_hash[r["task_set_hash"]].append(r)
    if not by_hash:
        out.append("_No results yet._")
    for h, rs in by_hash.items():
        out += [f"Task set `{h}`", "",
                "| model | type | n | mean score | 95% CI | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | incomplete (infra errors) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rs:
            out.append("| {model} | {typ} | {n} | {ms:.3f} | {ci} | {sr:.0%} | {fr:.0%} | {sc:.2f} | {lat:.2f} | {err} |".format(
                model=r["model"], typ="baseline" if r["baseline"] else "model", n=r["n"], ms=r["mean_score"],
                ci=f"+/-{r['ci95']:.3f}" if r.get("ci95") else "exact" if "exact" in (r["model"] or "") else "-",
                sr=r["solve_rate"], fr=r["feasible_rate"], sc=r["mean_sim_calls"], lat=r["mean_latency_s"],
                err=r["errors"]))
        out.append("")
    out.append("Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. "
               "n = tasks with a graded answer; rate-limit/provider failures are excluded and shown as incomplete. "
               "Baselines are not model inference. 95% CI = +/-1.96 standard errors over tasks (one run per task). "
               "Exact-expectation rows average a random baseline over every possible draw; the plain random and "
               "6-probe rows are one seeded draw.")
    return "\n".join(out) + "\n"


def write_report(results_dir, out_path, task_set_hash: str | None = None, tasks: list[dict] | None = None) -> dict:
    lb = build_leaderboard(results_dir, task_set_hash, tasks)
    Path(out_path).write_text(to_markdown(lb))
    return lb
