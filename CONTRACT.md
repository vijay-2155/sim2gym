# sim2gym internal contract (shared by the two build agents)

Project root: `/home/cheeku1855/dev/hacktoberfest` (Fedora Linux, NOT Windows; ignore BUILD_PLAN §2 machine facts).
Python: always `.venv/bin/python` (3.12, already has scalesim 3.0.0, numpy, openai, boto3, fastapi, uvicorn, pytest, httpx).
Package is installed editable from `src/sim2gym`. Do NOT `git commit`; the coordinator commits.
Spec: `BUILD_PLAN.md` (wins) > `CLAUDE_HANDOFF.md` > `PROJECT_CONTEXT.md` §4.2–4.5, §10.

## Ownership (do not edit the other side's files)

| Owner | Files |
|---|---|
| **Codex** (simulator core) | `src/sim2gym/packs/accel.py`, `src/sim2gym/tasks.py`, `src/sim2gym/grading.py`, `src/sim2gym/protocol.py`, `tests/test_accel.py`, `tests/test_tasks.py`, `tests/test_grading.py`, `tests/test_protocol.py` |
| **Claude agent** (runtime + API + live demo) | `src/sim2gym/env.py`, `src/sim2gym/baselines.py`, `src/sim2gym/models.py`, `src/sim2gym/runner.py`, `src/sim2gym/report.py`, `src/sim2gym/cli.py`, `src/sim2gym/api.py`, `src/sim2gym/static/*`, `tests/test_env.py`, `tests/test_runner.py`, `tests/test_api.py`, `tests/test_cli.py` |
| Coordinator | `pyproject.toml`, `README.md`, `__init__.py`, `.gitignore`, this file |

## Interfaces (exact names; both sides code against these)

### `sim2gym.packs.accel`  (wraps real SCALE-Sim)
```python
SCALESIM_VERSION: str                     # e.g. "3.0.0"
SIM_NAME = "accel"
ARRAY_DIMS: list[int]                     # e.g. [4, 8, 16, 32, 64, 128]
DATAFLOWS: list[str]                      # ["os", "ws", "is"]
SRAM_SPLITS: list[str]                    # e.g. ["balanced", "ifmap_heavy", "filter_heavy"] (may shrink to ["balanced"])
PE_BUDGETS: list[int]                     # e.g. [256, 1024, 4096]
WORKLOADS: dict[str, dict]                # name -> {"name", "kind": "gemm"|"conv", "source": str, plus SCALE-Sim layer fields}
def describe_layer(layer: dict) -> str    # human-readable shape text for prompts (no hints about best design)
def validate_design(design: dict, pe_budget: int) -> str | None
    # design keys exactly {"array_h","array_w","dataflow","sram_split"}; returns error string or None.
    # Rejects unknown/missing keys, non-int dims, values outside allowed sets, bools, h*w > pe_budget.
def simulate(layer: dict, array_h: int, array_w: int, dataflow: str, sram_split: str) -> dict
    # REAL SCALE-Sim run (disk-cached under cache/, key = hash(inputs + SCALESIM_VERSION)).
    # returns {"cycles": int, "stall_cycles": int, "utilization_pct": float}
def feasible_designs(pe_budget: int) -> list[dict]   # every valid design for that budget
```

### `sim2gym.tasks`
```python
def build_tasks(seed: int = 0, budget: int = 6, workloads: list[str] | None = None, pe_budgets: list[int] | None = None) -> list[dict]
def save_tasks(path, tasks) -> None ;  def load_tasks(path) -> list[dict]     # JSONL
def public_task(task: dict) -> dict      # strips "optimum" and anything else revealing the answer
```
Task dict (design tasks, one per workload × PE budget):
```json
{"id": "accel/design/<layer>/pe<budget>", "sim": "accel", "sim_version": "<scalesim ver + wrapper hash>",
 "kind": "design", "seed": 0, "layer_name": "...", "layer": {...}, "pe_budget": 1024,
 "space": {"array_h": [..], "array_w": [..], "dataflow": [..], "sram_split": [..]},
 "free": ["array_h","array_w","dataflow","sram_split"], "objective": {"minimize": "cycles"},
 "optimum": {"inputs": {"array_h":..,"array_w":..,"dataflow":..,"sram_split":..}, "value": <cycles int>, "method": "exhaustive"},
 "budget": 6, "prompt": "<rendered by protocol.render_prompt>"}
```

### `sim2gym.grading`
```python
def grade_design(task: dict, design) -> dict
    # {"score": float in [0,1], "solved": bool (score>=0.95), "feasible": bool, "cycles": int|None, "error": str|None}
    # invalid / over budget / non-dict -> score 0, feasible False. score = optimum_cycles / answer_cycles clipped.
```

### `sim2gym.protocol`
```python
def render_prompt(task: dict) -> str     # layer, choices + allowed values, PE budget, objective, protocol, sim budget. NO formulas/hints.
def parse_reply(text: str) -> dict | None
    # last valid top-level JSON object in text containing exactly one of "simulate" / "answer"; handles ```json fences,
    # nested objects, trailing junk, scientific notation; rejects NaN/Infinity. Returns {"simulate": {...}} or {"answer": {...}} or None.
RETRY_MESSAGE: str                       # sent once after a malformed reply
def format_observation(result: dict) -> str   # JSON text the env sends back after a simulate call
```

### Claude-agent side (for reference)
- `env.Sim2GymEnv(tasks, budget=6)`: `reset(i)->str prompt`, `step(text)->(obs, reward, done, info)`; max turns = budget+2; one malformed retry; terminal rejects steps.
- `models.py`: `bedrock:<modelId>` via boto3 `converse` in **us-east-1** (temperature 0) and `openai:<model>` via OpenAI-compatible base URL (Ollama). Model ids verified working 2026-11-01: `google.gemma-3-4b-it`, `google.gemma-3-12b-it`, `google.gemma-3-27b-it`, `qwen.qwen3-32b-v1:0`, `qwen.qwen3-next-80b-a3b`, `openai.gpt-oss-20b-1:0` / `-120b-1:0` (reasoning; needs larger maxTokens, text may be in reasoningContent), `mistral.mistral-large-3-675b-instruct`, `nvidia.nemotron-super-3-120b`, `deepseek.v3.2`.
- baselines: `oracle`, `random`, `max-square-ws`, `6-probe`.
