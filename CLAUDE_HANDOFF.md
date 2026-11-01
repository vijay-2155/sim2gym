# Claude handoff: build the sim2gym backend

## User instruction

The user wants a **working backend** for this hackathon project. They explicitly
redirected the task from demo planning to implementation, then requested this
handoff to Claude. Build and verify the backend; do not stop at another plan.
Do not build a frontend in this task.

## Current state

- Workspace: `/home/cheeku1855/dev/hacktoberfest`.
- Read `PROJECT_CONTEXT.md` first. It contains the product spec, five simulator
  formulas, parameter ranges, grading rules, architecture, and CLI examples.
- Before this handoff, the workspace contained only that spec and restricted
  metadata directories. **No application code, package, tests, or server exists.**
- Codex inspected the spec and environment; it did not implement the backend,
  run evaluations, start a server, or change `PROJECT_CONTEXT.md`.
- Python is 3.14.8. Module discovery found numpy, scipy, fastapi, uvicorn,
  pytest, openai, httpx, and setuptools. Imports, versions, and compatibility
  have not been tested. Declare dependencies in the package anyway.
- No model endpoint, installed model name, or API credentials have been verified.
  Model names mentioned in the spec are proposed choices, not confirmed availability.
- In Codex's environment only the workspace and `/tmp` were writable; metadata
  directories including `.git` were read-only. Check your own permissions.

## Product and scope

sim2gym wraps a deterministic Python simulator to produce seeded benchmark
tasks, code-based grading, and an interactive environment. ChipGym is the first
pack. The demo centers on an open model making chip-design choices, trying the
simulator within a budget, and receiving a score against an optimizer.

Implement the shared Python core, CLI, and an HTTP API a later web dashboard
can call. Use the smallest complete implementation; avoid duplicate evaluation
logic between API, CLI, and environment. FastAPI is available for the HTTP layer.

### Core

1. Public simulator API: `simulator`, `Param`, `Choice`, `Design`, `get_sim`,
   `list_sims`. Validate ranges, choices, integer inputs, physical validity,
   unknown keys, and non-finite numbers. Units must be explicit.
2. All five chip simulators from the spec: MOSFET saturation, oxide growth,
   CMOS power, chiplet cost, and LLM accelerator. Include one small non-chip
   simulator demonstrating the generic framework.
3. Seeded `predict`, `whatif`, and `design` task generation. Round sampled
   inputs before calculating answers. Record simulator version and task identity.
   Reject invalid samples and infeasible design tasks with a bounded retry limit.
4. Reliable optimizer references: enumerate integer choices, handle mixed inputs,
   and validate the final reference design. Use exact solutions where simple;
   label numerical references honestly rather than claiming a proven global optimum.
5. Grading: tolerance for prediction, exact what-if labels, zero for infeasible
   designs, and bounded objective-ratio scores for feasible designs. Guard zero
   denominators and non-finite values.
6. `Sim2GymEnv.reset()` and `step(text)` shared by evaluations and interactive
   sessions. Implement the spec's JSON answer/simulate protocol, five-call
   default budget, one malformed-answer retry, and bounded turns. Invalid
   attempts must not allow unlimited experiments; terminal sessions reject steps.
7. Oracle and seeded random baselines, explicitly labeled as baselines.
   Never present their results as model inference.
8. OpenAI-compatible model runner with endpoint/key configured server-side via
   environment variables, temperature zero, timeout, and clear failure results.
   Record model settings, calls, latency, errors, and actual transcripts.
   Never pass expected answers or reference designs to the model.
9. JSONL tasks/results, loading, and reports grouped by model, task type, and
   simulator. Report sample counts and experiment usage. Only compare scores
   fairly on matching task sets/settings; do not silently mix unrelated runs.
10. CLI: `sim2gym list`, `build`, `run`, `report`, and an API startup command
    or documented uvicorn command. Preserve the spec's intended command syntax
    where practical. Support custom simulator loading locally, not uploaded code.

### HTTP API

Use clear request/response schemas and generated `/docs`. Suggested routes:

- `GET /health`: service status; do not claim model readiness without checking it.
- `GET /simulators`: names, descriptions, units, ranges, and design metadata.
- `POST /simulators/{name}/simulate`: validate inputs and return real outputs.
- `POST /tasks`: generate a seeded task set with a server-side identifier.
- `GET /tasks/{id}`: retrieve public task data, excluding expected answers and
  optimizer references from the participant view.
- `POST /sessions`: start an environment for a stored task.
- `POST /sessions/{id}/step`: accept the protocol text and return observation,
  reward, done, experiment usage, and final scoring details when complete.
- `POST /runs`: evaluate stored tasks using oracle, random, or a configured model.
- `GET /runs/{id}` and `GET /leaderboard`: retrieve real stored results/reports.

These route names are suggestions; document the final contract. A synchronous
run endpoint is sufficient initially if bounded and documented. Preserve run
results on disk. In-memory interactive sessions are acceptable for a local demo
if restart behavior and single-worker operation are documented. Serialize access
to mutable sessions and result files if handling concurrent requests.

Validate names, IDs, task counts, budgets, and request sizes. Do not accept
arbitrary filesystem paths, Python source, or model endpoint URLs from HTTP
clients. Keep keys out of logs, API responses, and saved results. Bind to
localhost by default. Explain that this demo API has no public multi-user auth.

## Build order

First get the accelerator task working end to end through the same shared core:
simulator → generated task → environment → oracle runner → saved result → API.
Then complete the remaining simulators, task types, model integration, and CLI.
Deliver the whole authorized backend scope, with tests and a README.

## Acceptance checks

- A clean environment can install the package using the documented commands.
- All five simulators pass independently calculated physics checks from the
  spec. Check questionable constants/expected values rather than adjusting
  implementations merely to satisfy an approximate number in the spec.
- Identical seeds/settings reproduce tasks; displayed inputs reproduce answers.
- Oracle scores 1 on every generated task across multiple seeds and simulators.
- Wrong and infeasible answers score 0; random scores are measured, not assumed
  to approach zero on categorical tasks or easy feasibility problems.
- Test malformed/nested/trailing JSON, scientific notation, NaN/infinity,
  integer constraints, simulation budgets, retries, and terminal behavior.
- HTTP integration test creates a task, starts a session, simulates a candidate,
  submits an answer, and retrieves its result. Check invalid requests and that
  public task responses do not leak answers/references.
- Exercise the model runner against a local fake OpenAI-compatible HTTP server
  to test an experiment, final answer, timeout, and failure without credentials.
  Label this as protocol testing, never as a real open-model evaluation.
- Smoke-test CLI build → oracle/random run → report and an actual HTTP server.
- If a real open-model endpoint is available, run and save a small real eval.
  Otherwise state exactly what remains unverified; do not fabricate a leaderboard
  or claim the hackathon requirement of two open models is satisfied.

## Deliverables and constraints

Deliver installable source, tests, README quickstart, environment-variable
example without secrets, API examples, local demo commands, and MIT license as
specified. Keep original project context intact and note any material deviations.
No frontend, login system, GPU training, cloud deployment, SPICE/PDK installation,
or fabricated AI findings are needed for this backend task.

The simulators are simplified textbook models, not production EDA or measured
hardware. An RL environment interface is not an actual RL training run.

User prefers direct action and concise progress updates. Finish with what works,
how to run it, checks actually completed, and remaining limitations.
