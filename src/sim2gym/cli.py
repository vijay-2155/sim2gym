"""sim2gym CLI: list | build | run | report | serve | models | quota."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_list(args) -> int:
    from sim2gym.baselines import BASELINES
    from sim2gym.models import BEDROCK_MODELS
    from sim2gym.packs import accel
    print(f"simulator: {accel.SIM_NAME} (SCALE-Sim {accel.SCALESIM_VERSION})")
    print(f"  array dims {accel.ARRAY_DIMS}, dataflows {accel.DATAFLOWS}, sram splits {accel.SRAM_SPLITS}, "
          f"PE budgets {accel.PE_BUDGETS}")
    print("workloads:")
    for name, layer in accel.WORKLOADS.items():
        print(f"  {name}  [{layer['kind']}]")
    print("baselines: " + ", ".join(BASELINES))
    print("bedrock models (paid, short tests): " + ", ".join(f"bedrock:{m}" for m in BEDROCK_MODELS))
    print("openrouter free models: see `sim2gym models --free`")
    return 0


def cmd_build(args) -> int:
    from sim2gym import tasks as tasks_mod
    from sim2gym.runner import task_set_hash
    wl = args.workloads.split(",") if args.workloads else None
    pe = [int(x) for x in args.pe_budgets.split(",")] if args.pe_budgets else None
    tasks = tasks_mod.build_tasks(seed=args.seed, budget=args.budget, workloads=wl, pe_budgets=pe)
    tasks_mod.save_tasks(args.output, tasks)
    print(f"wrote {len(tasks)} tasks to {args.output} (task_set_hash {task_set_hash(tasks)})")
    return 0


def cmd_run(args) -> int:
    from sim2gym import runner
    tasks = runner.load_tasks(args.tasks)
    ids = args.task_ids.split(",") if args.task_ids else None

    def show(r):
        tag = "ERROR " + str(r["error"])[:80] if r["infra_error"] else (r["error"] or "")
        print(f"  {r['task_id']}: score {r['score']:.3f} sims {r['sim_calls_used']} turns {r['turns']} "
              f"lat {r['latency_s']:.1f}s {tag}", flush=True)

    summary = runner.run(tasks, args.model, results_dir=args.results, workers=args.workers, budget=args.budget,
                         limit=args.limit, task_ids=ids, resume=not args.no_resume, on_result=show,
                         max_tokens=args.max_tokens, timeout_s=args.timeout, seed=args.seed,
                         allow_paid=args.allow_paid)
    if summary.get("paid_guard"):
        print(f"NOTE: paid model guard active: max {runner.PAID_GUARD_MAX_TASKS} tasks, max_turns "
              f"{runner.PAID_GUARD_MAX_TURNS}, results in results/smoke/ (pass --allow-paid for a full run)",
              file=sys.stderr)
    if summary.get("rate_limited"):
        print("WARNING: rate limited; stopped early. Re-run later to resume.", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_models(args) -> int:
    from sim2gym import models
    from sim2gym.baselines import BASELINES
    free = models.openrouter_free_models(force=True)
    if models.free_models_error():
        print("WARNING: could not list OpenRouter models: " + models.free_models_error(), file=sys.stderr)
    print(f"OpenRouter free models ({len(free)}):")
    for m in free:
        print(f"  openrouter:{m['id']}  ctx {m.get('context_length')}")
    if not args.free:
        print("baselines: " + ", ".join(BASELINES))
        print("bedrock (paid, short tests): " + ", ".join(f"bedrock:{m}" for m in models.BEDROCK_MODELS))
    d = models.default_free_model(free)
    print(f"default demo model: {d or models.FALLBACK_DEMO_MODEL}")
    return 0


def cmd_quota(args) -> int:
    from sim2gym import models
    q = models.openrouter_quota()
    print(json.dumps(q, indent=2))
    return 1 if "error" in q else 0


def cmd_report(args) -> int:
    from sim2gym import report
    h = None
    if args.tasks:
        from sim2gym.runner import load_tasks, task_set_hash
        h = task_set_hash(load_tasks(args.tasks))
    lb = report.write_report(args.results_dir, args.output, task_set_hash=h)
    for w in lb["warnings"]:
        print("WARNING: " + w, file=sys.stderr)
    print(open(args.output).read())
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("sim2gym.api:app", host=args.host, port=args.port, workers=1)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sim2gym", description="Real simulators as budgeted black-box RL environments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="simulator, workloads, baselines, models")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("build", help="build design tasks with exhaustive optima")
    s.add_argument("pack", nargs="?", default="accel", choices=["accel"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--budget", type=int, default=6)
    s.add_argument("--workloads", help="comma-separated workload names (default all)")
    s.add_argument("--pe-budgets", help="comma-separated PE budgets (default all)")
    s.add_argument("-o", "--output", default="tasks.jsonl")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("run", help="run a model or baseline over tasks")
    s.add_argument("tasks")
    s.add_argument("--model", required=True,
                   help="oracle | random | max-square-ws | 6-probe | openrouter:<id> | bedrock:<modelId> | openai:<model>")
    s.add_argument("--workers", type=int, default=None, help="threads (default 4 for bedrock, else 1)")
    s.add_argument("--budget", type=int, default=None, help="override simulator-call budget (default: task's)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--task-ids", default=None, help="comma-separated task ids")
    s.add_argument("--results", default="results")
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--timeout", type=float, default=None)
    s.add_argument("--seed", type=int, default=0, help="seed for random baselines")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--allow-paid", action="store_true",
                   help="lift the paid-model guard (bedrock: default max 2 tasks, max_turns 2)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("models", help="list OpenRouter free models (cached 10 min), baselines, Bedrock")
    s.add_argument("--free", action="store_true", help="only OpenRouter free models")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("quota", help="OpenRouter key limit/usage (never prints the key)")
    s.set_defaults(fn=cmd_quota)

    s = sub.add_parser("report", help="write leaderboard markdown")
    s.add_argument("results_dir", nargs="?", default="results")
    s.add_argument("-o", "--output", default="leaderboard.md")
    s.add_argument("--tasks", default=None, help="only include results for this tasks file's task set")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("serve", help="HTTP API + live demo (localhost)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
