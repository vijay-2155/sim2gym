# sim2gym: real simulators → black-box RL environments for open models

> Chip companies make accelerator trade-offs every day. **sim2gym wraps a real open-source accelerator
> simulator as an RL environment.** The model has to *experiment* rather than recall a formula. This repo shows how
> open models do, and where they fail.

**AccelGym** is the first environment. The model designs a systolic-array AI accelerator for a real neural-network
layer:

- **Choices:** array height × width and dataflow (`os` / `ws` / `is`)
- **Constraint:** a processing-element (PE) budget
- **Goal:** minimise cycles

The model never sees a formula. It may call the **real SCALE-Sim simulator** up to 6 times. Its final answer is
then graded by SCALE-Sim against an **exact optimum** found by exhaustive search.

Built on open source:

| Project | Role |
|---|---|
| [SCALE-Sim](https://github.com/scalesim-project/scale-sim-v2) (`scalesim` 3.0.0, Georgia Tech) | Real cycle-level systolic-array simulator; grades every answer |
| Amazon Bedrock open-weight models (Nemotron, Qwen3, gpt-oss, Mistral, DeepSeek, Gemma 3) | Models evaluated on the leaderboard |
| [OpenRouter](https://openrouter.ai) free open-weight models (Nemotron, Gemma 4, …) | Free backend (rate-limited free tier) |
| Ollama / any OpenAI-compatible server | Optional local backend |
| Python, NumPy, FastAPI | Core, API and live demo |

## Quickstart

```bash
uv venv -p python3.12 .venv          # SCALE-Sim 3.0.0 works on 3.12
uv pip install -p .venv/bin/python -e '.[dev]'
cp .env.example .env                 # put OPENROUTER_API_KEY=... in .env (gitignored)

.venv/bin/sim2gym list                               # simulator, workloads, design space
.venv/bin/sim2gym build --seed 0 -o tasks.jsonl      # 36 design tasks with exact optima (fills cache/)
.venv/bin/sim2gym run tasks.jsonl --model oracle     # sanity check: must score 1.000
.venv/bin/sim2gym run tasks.jsonl --model 6-probe    # the baseline a model must beat
.venv/bin/sim2gym models --free                      # live list of free OpenRouter models
.venv/bin/sim2gym run tasks_eval.jsonl --model openrouter:nvidia/nemotron-3.5-super-vl:free --results results_eval
.venv/bin/sim2gym report results_eval --tasks tasks_eval.jsonl -o leaderboard_eval.md
.venv/bin/sim2gym serve                              # live demo at http://127.0.0.1:8765/  (API docs: /docs)
.venv/bin/python -m pytest -q                        # 125 tests
```

The first `build` runs SCALE-Sim over the whole grid: 576 simulations, about 3.5 minutes with 6 workers. After that
every simulate call is a cache hit, keyed by the input hash plus the SCALE-Sim version.

## Live demo

`sim2gym serve`, then open http://127.0.0.1:8765/. Pick a task (layer × PE budget) and a model, then press **Run**.
The page streams the episode over Server-Sent Events:

- every model message
- every SCALE-Sim call (array, dataflow, cycles, utilisation) against the budget meter
- the final design next to the exhaustive optimum, with its score

The **Leaderboard** tab reads `results/`.

![Live episode](docs/screenshots/live-episode.png)
![Leaderboard](docs/screenshots/leaderboard.png)

## The environment

| | |
|---|---|
| Workloads | 12 real layers: 9 BERT-tiny GEMMs (attention projection, MLP up and MLP down at sequence lengths 8/32/128) and 3 ResNet-18 / MobileNetV2 1×1 convolutions. Tall-skinny and wide shapes are included on purpose. |
| Design space | `array_h`, `array_w` ∈ {8, 16, 32, 64}; `dataflow` ∈ {os, ws, is}; SRAM fixed at 3 × 64 kB; bandwidth fixed |
| PE budgets | 256 / 1024 / 4096, giving 18 / 39 / 48 feasible designs |
| Tasks | 36 (12 layers × 3 budgets). In 24 of them the optimal array is **non-square**. |
| Protocol | The model ends every reply with one JSON object: `{"simulate": {...}}` costs 1 of 6 calls and returns cycles, utilisation and stalls; `{"answer": {...}}` ends the episode. One malformed-reply retry; max turns = budget + 2. |
| Grading | Infeasible, invalid or over the PE budget → 0. Otherwise `score = optimum_cycles / answer_cycles`, which is in [0, 1]. Solved = score ≥ 0.95. |

The optimum comes from an exhaustive search over a discrete space, so it is exact, not a heuristic.

### RL / programmatic use

```python
from sim2gym.tasks import load_tasks
from sim2gym.env import Sim2GymEnv

env = Sim2GymEnv(load_tasks("tasks.jsonl"))
obs = env.reset(0)                                   # task prompt
obs, reward, done, info = env.step('{"simulate": {"array_h": 16, "array_w": 64, "dataflow": "os", "sram_split": "balanced"}}')
# reward stays 0 until the episode ends, then equals the task score in [0, 1]
```

The evaluation runner, the baselines and the live demo all drive this same `Sim2GymEnv`, so evaluation and RL share
one scoring path.

## Results

All 36 tasks (`tasks.jsonl`, task set `903fb5045d2d4d26`), on Amazon Bedrock open-weight models at temperature 0.
Every number below comes from `results/*.jsonl` (see [`leaderboard.md`](leaderboard.md)).

| Model | Type | n | Mean score | Solved (≥0.95) | Feasible | Mean sim calls | Mean latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| oracle (exhaustive optimum) | baseline | 36 | 1.000 | 100% | 100% | 0.00 | – |
| **Nemotron Super 3 120B** (`nvidia.nemotron-super-3-120b`) | model | 36 | **0.838** | 56% | 100% | 5.00 | 5.3 |
| **Qwen3 32B** (`qwen.qwen3-32b-v1:0`) | model | 36 | **0.804** | 53% | 100% | 5.06 | 7.4 |
| Gemma 3 27B (`google.gemma-3-27b-it`) | model | 36 | 0.709 | 36% | 100% | 4.97 | 34.0 |
| gpt-oss 120B (`openai.gpt-oss-120b-1:0`) | model | 36 | 0.706 | 36% | 100% | 3.11 | 11.2 |
| DeepSeek V3.2 (`deepseek.v3.2`) | model | 36 | 0.704 | 36% | 100% | 5.11 | 68.9 |
| **6-probe** (6 random simulator calls, keep the best) | baseline | 36 | 0.650 | 28% | 100% | 6.00 | – |
| Mistral Large 3 675B (`mistral.mistral-large-3-675b-instruct`) | model | 36 | 0.590 | 22% | 94% | 0.42 | 5.4 |
| max-square-ws (largest square array, ws dataflow) | baseline | 36 | 0.521 | 8% | 100% | 0.00 | – |
| random feasible design | baseline | 36 | 0.380 | 8% | 100% | 0.00 | – |

All six models ran on the same 36 tasks with the same settings.

### What the results show

- **Experimenting beats size.** Every model that used the simulator budget (about 5 calls per task) beat the
  6-probe baseline. Mistral Large 3 answered **33 of 36 tasks without a single simulator call** and scored below
  6-probe. 2 of its answers used array sizes that are not allowed, so they score 0. gpt-oss 120B used about 3 calls
  and landed in between.
- **A "bigger array is better" bias.** In 12–17 of the 36 tasks, each model's final array was larger than the
  optimal one.

### A real failure (from `results/bedrock_qwen.qwen3-32b-v1_0.jsonl`)

**Task:** `bert_tiny_mlp_down_s8`, a GEMM (8×512)·(512×128) with a 4096-PE budget. The exhaustive optimum is
**8×8, os: 8,415 cycles at 97% utilisation**.

| Qwen3 32B call | Design | SCALE-Sim cycles | Utilisation |
|---|---|---:|---:|
| 1 | 64×64 os | 13,231 | 1.0% |
| 2 | 32×64 os | 13,167 | 1.9% |
| 3 | 16×64 os | 13,135 | 3.9% |
| 4 | 8×64 os | 13,119 | 7.8% |
| 5 | 64×64 ws | 13,107 | 1.0% |
| **answer** | **64×64 ws** | **13,107** | score **0.64** |

The model shrank only the height and never tried a narrow array, although utilisation was about 1%. Mistral Large 3
answered 32×64 ws on this task without simulating. On ResNet-18's `layer4` downsample, Qwen3 found the tall-skinny
optimum **64×8** by experiment (score 1.0). Mistral guessed 64×32 without simulating and scored 0.40.

### Free models (OpenRouter)

The OpenRouter backend works and was tested live, but free-tier keys are capped at 50 free-model requests per day.
Our 12-task run of Nemotron 3.5 Super, Nemotron 3 Ultra, dots-3-note and Gemma 4 hit that cap after 1–3 tasks per
model. Those partial results are in `results_eval/` and are not ranked.

## Wrap your own simulator

The environment only needs a plain function. `src/sim2gym/packs/accel.py` is the whole SCALE-Sim integration:

- write a config and a topology file
- run `scalesim`
- parse `COMPUTE_REPORT.csv`
- cache the result

```python
def simulate(layer: dict, array_h: int, array_w: int, dataflow: str, sram_split: str) -> dict:
    ...  # run the real tool
    return {"cycles": ..., "stall_cycles": ..., "utilization_pct": ...}
```

To add another tool, write the same function plus `feasible_designs()` and `validate_design()`. Tasks, exact
optima, grading, the environment, the runner, the API and the live page all reuse them.

## HTTP API

`sim2gym serve` binds to 127.0.0.1:8765. The schemas are documented at `/docs`.

| Route | Purpose |
|---|---|
| `GET /health` | Service status (does not claim the model endpoints are reachable) |
| `GET /simulators`, `POST /simulators/accel/simulate` | Design space metadata; real SCALE-Sim run |
| `GET /models` | Free OpenRouter models, baselines and Bedrock models (allowlisted server-side) |
| `POST /tasks`, `GET /tasks/{id}` | Seeded task sets; the public view never includes the optimum |
| `POST /sessions`, `POST /sessions/{id}/step` | Interactive episode using the JSON protocol |
| `POST /runs`, `GET /runs/{id}`, `GET /leaderboard` | Bounded synchronous runs; leaderboard from saved results |
| `GET /live?task_id=…&model=…` | Server-Sent Events stream of one episode (used by the demo page) |

Model keys and endpoints are server-side only: the key comes from `.env` or the environment and is never logged,
returned or saved. API clients cannot supply URLs, file paths or code. This is a **single-user local demo**: there
is no auth, sessions live in memory (lost on restart), and it needs a single worker.

## Model backends

| Prefix | Backend | Notes |
|---|---|---|
| `openrouter:<id>` | OpenRouter (main) | Free models (`:free`); temperature 0; max 12k tokens, because reasoning models think first; 429s back off and honour Retry-After |
| `bedrock:<modelId>` | Amazon Bedrock Converse (us-east-1, AWS CLI credentials) | **Paid:** by default capped at 2 tasks × 2 turns and written to `results/smoke/`; `--allow-paid` (CLI) or `SIM2GYM_ALLOW_PAID=1` (server) lifts the cap |
| `openai:<model>` | Any OpenAI-compatible server (e.g. Ollama) | Uses `SIM2GYM_BASE_URL` |
| `oracle`, `random`, `max-square-ws`, `6-probe` | Baselines, labelled as such | Not model inference |

## Honest limits

- v0.1, built in one day for Hacktoberfest Hack Day × AI for Vizag.
- One real simulator so far, with a small discrete design space (4 × 4 × 3, SRAM split fixed) and small layers so
  that simulation stays fast.
- SCALE-Sim models cycles, not area or power; the PE budget stands in for area. The reported cycles cover compute
  plus stalls, excluding the initial prefetch and the final DRAM drain.
- Evaluated on hosted open-weight models at temperature 0, one run each, 36 tasks. That is enough to see patterns,
  not to claim fine-grained rankings. No RL training run yet.
- Prior work: LUMINA (Mar 2026) and PerfReasoning (Sep 2026) already test LLMs on hardware trade-offs. sim2gym's
  contribution is a **generator of black-box, budgeted environments from real simulators**, not "the first
  chip-reasoning benchmark".
- Next steps: a `verifiers` / Environments Hub adapter, ngspice + SKY130 inverter sizing, and RL training.

## License

MIT
