<div align="center">

# sim2gym

**Turn real simulators into budgeted, black-box RL environments and benchmarks for language models.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-155%20passing-brightgreen.svg)](#development)
[![Simulator: SCALE-Sim 3.0.0](https://img.shields.io/badge/simulator-SCALE--Sim%203.0.0-orange.svg)](https://github.com/scalesim-project/scale-sim-v2)
[![Live results](https://img.shields.io/badge/live-results-black.svg)](https://sim2gym-results.vercel.app)

[Live results](https://sim2gym-results.vercel.app) · [Quickstart](#quickstart) · [Results](#results) ·
[How it works](#how-it-works) · [Contributing](#contributing)

</div>

---

sim2gym wraps a real engineering simulator as an environment where a model has to **experiment** to find a good
answer. Recalling a formula is not enough. Every answer is graded by the real simulator against an exact optimum
found by exhaustive search.

The first environment, **AccelGym**, asks a model to design a systolic-array AI accelerator for a real
neural-network layer. The model may run the open-source [SCALE-Sim](https://github.com/scalesim-project/scale-sim-v2)
simulator up to 6 times before it commits to a design.

<p align="center"><img src="docs/screenshots/insights.png" alt="sim2gym Insights view" width="820"></p>

## Highlights

- **Real simulator, exact grading.** Each task's optimum comes from simulating every feasible design, so scores are
  exact, not heuristic.
- **Budgeted and black-box.** The model sees only simulator outputs (cycles, utilisation, stalls) and has 6 calls.
- **Unlimited unseen tasks.** A seeded generator builds new layers with the same exact optimum, so a held-out split
  can't have been memorised.
- **One scoring path for evaluation and RL.** The CLI runner, the baselines, the HTTP API, the live demo and a
  Prime Intellect [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) adapter all use the same
  `Sim2GymEnv`.
- **Many model backends.** Amazon Bedrock, OpenRouter, any OpenAI-compatible server (Ollama), and the Kiro and Codex
  agent CLIs.
- **Honest statistics.** 95% confidence intervals, paired comparisons, and random baselines reported as exact
  expectations rather than one lucky draw.

## Table of contents

- [Quickstart](#quickstart)
- [Results](#results)
- [How it works](#how-it-works)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Project structure](#project-structure)
- [Extending sim2gym](#extending-sim2gym)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Development](#development)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Quickstart

**Requirements:** Python 3.12 (SCALE-Sim 3.0.0 is tested on it) and [uv](https://docs.astral.sh/uv/). Linux or macOS
is needed for the optional `verifiers` adapter.

```bash
git clone https://github.com/vijay-2155/sim2gym.git
cd sim2gym
uv venv -p python3.12 .venv
uv pip install -p .venv/bin/python -e '.[dev]'
cp .env.example .env            # optional: model API keys (gitignored, never logged)
```

The SCALE-Sim results for both task sets are checked in under `cache/`, so the following run instantly, with no
simulation and no API keys:

```bash
.venv/bin/sim2gym run tasks.jsonl --model oracle     # sanity check: must score 1.000
.venv/bin/sim2gym run tasks.jsonl --model 6-probe    # the random-search baseline a model must beat
.venv/bin/sim2gym serve                              # demo at http://127.0.0.1:8765  (API docs at /docs)
```

To evaluate a model, pick a [backend](#model-backends):

```bash
.venv/bin/sim2gym run tasks.jsonl --model openai:qwen3:4b                          # local Ollama
.venv/bin/sim2gym run tasks.jsonl --model bedrock:qwen.qwen3-32b-v1:0 --allow-paid # Amazon Bedrock
.venv/bin/sim2gym report results --tasks tasks.jsonl -o leaderboards/fixed.md
```

## Results

**Browse them at [sim2gym-results.vercel.app](https://sim2gym-results.vercel.app).** Every number here is computed
from the per-task records in `results/` and `results_gen/`. The full tables are in
[`leaderboards/fixed.md`](leaderboards/fixed.md) and [`leaderboards/generated.md`](leaderboards/generated.md).

### Fixed benchmark: 36 tasks (`tasks.jsonl`)

The baseline to beat is **random search**: 6 random simulator calls, keeping the best design. The table shows its
exact expectation over every possible draw, **0.723**. The last column is the model's per-task gain over random
search, with a paired 95% CI. ✅ means that interval is entirely above zero.

| Model | Access | Score | 95% CI | Solved ≥0.95 | Sim calls | vs random search (paired 95% CI) |
|---|---|---:|---:|---:|---:|---|
| oracle (exhaustive optimum) | baseline | 1.000 | – | 100% | – | |
| **GPT-6.1 Sol** | Codex CLI | **0.861** | ±0.067 | 64% | 5.67 | ✅ +0.138 [+0.066, +0.209] |
| GPT-6.1 Terra | Codex CLI | 0.838 | ±0.079 | 61% | 5.61 | ✅ +0.115 [+0.033, +0.197] |
| Nemotron Super 3 120B | Bedrock | 0.838 | ±0.071 | 56% | 5.00 | ✅ +0.115 [+0.040, +0.190] |
| Claude Opus 5.5 | Kiro CLI | 0.825 | ±0.070 | 56% | 4.17 | ✅ +0.101 [+0.020, +0.183] |
| Qwen3 32B | Bedrock | 0.804 | ±0.084 | 53% | 5.06 | ✅ +0.080 [+0.003, +0.158] |
| Gemma 4 E4B (local, Q4) | Ollama | 0.726 | ±0.085 | 33% | 5.67 | +0.002 [−0.077, +0.081] |
| *random search, 6 calls* | baseline | *0.723* | exact | 31% | 6.00 | |
| Gemma 3 27B | Bedrock | 0.709 | ±0.094 | 36% | 4.97 | −0.015 [−0.099, +0.070] |
| gpt-oss 120B | Bedrock | 0.706 | ±0.085 | 36% | 3.11 | −0.018 [−0.094, +0.059] |
| DeepSeek V3.2 | Bedrock | 0.704 | ±0.091 | 36% | 5.11 | −0.020 [−0.107, +0.067] |
| GPT-6 Luna | Codex CLI | 0.618 | ±0.112 | 36% | 1.94 | −0.106 [−0.217, +0.006] |
| Mistral Large 3 675B | Bedrock | 0.590 | ±0.102 | 22% | 0.42 | −0.133 [−0.233, −0.033] |
| max-square-ws (rule of thumb) | baseline | 0.521 | ±0.086 | 8% | – | |
| *random design, no simulation* | baseline | *0.399* | exact | 6% | – | |

### Unseen tasks: 36 generated tasks (`tasks_gen_s1.jsonl`, seed 1)

These layers were sampled by the generator and no model has seen them before. Random search scores **0.789** here,
so this set is slightly easier than the fixed one.

| Model | Access | Score | 95% CI | Solved ≥0.95 | Sim calls | vs random search (paired 95% CI) |
|---|---|---:|---:|---:|---:|---|
| **Claude Opus 5.5** | Kiro CLI | **0.942** | ±0.050 | 83% | 3.78 | ✅ +0.153 [+0.083, +0.223] |
| Nemotron Super 3 120B | Bedrock | 0.910 | ±0.052 | 61% | 5.00 | ✅ +0.122 [+0.061, +0.183] |
| DeepSeek V3.2 | Bedrock | 0.866 | ±0.060 | 50% | 5.17 | ✅ +0.078 [+0.015, +0.141] |
| Gemma 3 27B | Bedrock | 0.816 | ±0.073 | 42% | 4.78 | +0.027 [−0.044, +0.098] |
| Qwen3 32B | Bedrock | 0.803 | ±0.101 | 53% | 4.97 | +0.015 [−0.087, +0.116] |
| *random search, 6 calls* | baseline | *0.789* | exact | 36% | 6.00 | |
| gpt-oss 120B | Bedrock | 0.704 | ±0.096 | 31% | 2.81 | −0.085 [−0.194, +0.025] |
| Mistral Large 3 675B | Bedrock | 0.549 | ±0.112 | 25% | 0.78 | −0.240 [−0.349, −0.131] |
| max-square-ws (rule of thumb) | baseline | 0.464 | ±0.086 | 6% | – | |
| *random design, no simulation* | baseline | *0.444* | exact | 7% | – | |

All runs use temperature 0 where the backend allows it, with one run per task. Models closer than about 0.1 are not
reliably ranked; use the paired column for comparisons.

### Key findings

1. **Only 5 of 11 models clearly beat random search** on the fixed benchmark: GPT-6.1 Sol and Terra, Nemotron Super
   3, Claude Opus 5.5 and Qwen3 32B. Five more are statistically tied with it, and Mistral Large 3 is clearly
   below it.
2. **Skipping experiments is the costliest mistake.** Mistral Large 3 answered 33 of 36 tasks without a single
   simulator call, and GPT-6 Luna answered 24 of 36 that way. They have the two lowest scores of all 11 models.
3. **Running experiments is necessary, not sufficient.** DeepSeek V3.2 and Gemma 4 E4B used about 5–6 calls per
   task and still tied random search on the fixed set: the model also has to learn from what it measured.
4. **Strong models stop too early.** Claude Opus 5.5 was always right when it stopped after 1–3 calls (10 tasks,
   score 1.000). It averaged 0.709 on the 10 tasks where it stopped at 4 calls with 2 calls unused.
5. **Models over-build.** On the fixed set, every model's final array was larger than the optimum in 10–17 of the
   36 tasks. In 24 of the 36 tasks the best array is a narrow rectangle, not a square.
6. **Size isn't skill.** Nemotron Super 3 (120B) beat Mistral Large 3 (675B) by 0.25 on the fixed set and 0.36 on
   the generated set. A quantised Gemma 4 E4B running on a laptop matched Gemma 3 27B on Bedrock.

<details>
<summary><b>Example failure: Qwen3 32B on <code>bert_tiny_mlp_down_s8</code></b></summary>

The task is a GEMM (8×512)·(512×128) with a 4096-PE budget. The exhaustive optimum is **8×8 `os`: 8,415 cycles
at 97% utilisation**. Source: `results/bedrock_qwen.qwen3-32b-v1_0.jsonl`.

| Call | Design | SCALE-Sim cycles | Utilisation |
|---|---|---:|---:|
| 1 | 64×64 os | 13,231 | 1.0% |
| 2 | 32×64 os | 13,167 | 1.9% |
| 3 | 16×64 os | 13,135 | 3.9% |
| 4 | 8×64 os | 13,119 | 7.8% |
| 5 | 64×64 ws | 13,107 | 1.0% |
| **answer** | **64×64 ws** | **13,107** | score **0.64** |

The model shrank only the height and never tried a narrow array, although utilisation stayed near 1%.

</details>

## How it works

```text
          task prompt (layer shape, PE budget, design space)
                              │
                              ▼
   ┌───────────┐  {"simulate": {...}}   ┌──────────────────┐
   │   model   │ ─────────────────────▶ │ SCALE-Sim (real) │   up to 6 calls
   │           │ ◀───────────────────── │                  │
   └───────────┘  cycles, util, stalls  └──────────────────┘
         │
         │ {"answer": {...}}
         ▼
   grade = optimum_cycles / answer_cycles     (optimum from exhaustive search)
```

| | |
|---|---|
| **Workloads** | 12 real layers: 9 BERT-tiny GEMMs (attention projection, MLP up and MLP down at sequence lengths 8, 32 and 128) and 3 ResNet-18 / MobileNetV2 1×1 convolutions. |
| **Design space** | `array_h`, `array_w` ∈ {8, 16, 32, 64}; `dataflow` ∈ {`os`, `ws`, `is`}. SRAM is fixed at 3 × 64 kB and bandwidth is fixed. |
| **PE budgets** | 256, 1024 and 4096, giving 18, 39 and 48 feasible designs. |
| **Tasks** | 36 (12 layers × 3 budgets). In 24 of them the optimal array is non-square. |
| **Protocol** | Every reply ends with one JSON object. `{"simulate": {...}}` spends 1 of 6 calls and returns cycles, utilisation and stalls; `{"answer": {...}}` ends the episode. One malformed-reply retry is allowed, and the turn limit is the budget + 2. |
| **Grading** | An infeasible, invalid or over-budget design scores 0. Otherwise `score = optimum_cycles / answer_cycles`, in [0, 1]. A task counts as solved at ≥ 0.95. |
| **Baselines** | `oracle` (the optimum), `6-probe` (random search), `random`, and `max-square-ws` (largest square array, weight-stationary). |

The prompt never contains the optimum, and the public task view served by the API never includes it.

## Usage

### CLI

| Command | Description |
|---|---|
| `sim2gym list` | Show the simulator, workloads, design space, baselines and models. |
| `sim2gym build --seed 0 -o tasks.jsonl` | Build the fixed task set with exhaustive optima (576 simulations on a cold cache). |
| `sim2gym generate --seed 1 --n 12 -o tasks_gen.jsonl` | Sample `n` new layers and build `3n` tasks with exact optima. |
| `sim2gym run <tasks> --model <spec>` | Run a model or baseline over a task set. It resumes by default; `--workers`, `--limit`, `--task-ids`, `--results`. |
| `sim2gym report <results> --tasks <tasks>` | Write a Markdown leaderboard with CIs and exact baseline rows. |
| `sim2gym site -o site` | Build the static results site (Insights + Leaderboard as JSON, no server). |
| `sim2gym serve` | Start the HTTP API and live demo on `127.0.0.1:8765`. |
| `sim2gym models` / `sim2gym quota` | List free OpenRouter models / show OpenRouter key usage (the key itself is never printed). |

### Model backends

| Spec | Backend | Notes |
|---|---|---|
| `bedrock:<modelId>` | Amazon Bedrock Converse | Uses AWS CLI credentials, region `SIM2GYM_BEDROCK_REGION` (default `us-east-1`). **Paid:** capped at 2 tasks × 2 turns (written to `results/smoke/`) unless you pass `--allow-paid`. |
| `openrouter:<id>` | OpenRouter | `OPENROUTER_API_KEY`. Free models end in `:free`; 429s back off and honour `Retry-After`. |
| `openai:<model>` | Any OpenAI-compatible server | `SIM2GYM_BASE_URL` (default Ollama, `http://localhost:11434/v1`) and `SIM2GYM_API_KEY`. |
| `kiro:<model>` | Kiro CLI (e.g. `claude-opus-5.5`) | Uses your logged-in Kiro account and credits. Treated as paid. |
| `codex:<model>` | Codex CLI (e.g. `gpt-6.1-sol`) | Uses your logged-in ChatGPT plan. Treated as paid. |
| `oracle`, `6-probe`, `random`, `max-square-ws` | Baselines | Not model inference; labelled as baselines everywhere. |

The `kiro:` and `codex:` backends run the CLI once per turn: a fresh, stateless process in an empty temporary
directory, with every tool disabled (Kiro: no trusted tools; Codex: shell, browser and apps disabled, read-only
sandbox, no user config). The model therefore can't read `tasks.jsonl`, which contains the optima. Each CLI adds its
own agent system prompt, so these results are labelled with the CLI prefix and aren't identical to a raw API run.

### Live demo and HTTP API

`sim2gym serve` starts a FastAPI app with a single-page demo. The **Insights** view explains the project and the
findings in plain language. **Live episode** streams a run over Server-Sent Events: every model message, every
SCALE-Sim call, and the final design next to the optimum. **Leaderboard** reads `results/`.

<p align="center"><img src="docs/screenshots/live-episode.png" alt="Live episode view" width="820"></p>

| Route | Purpose |
|---|---|
| `GET /health` | Service status. |
| `GET /simulators`, `POST /simulators/accel/simulate` | Design-space metadata; a real SCALE-Sim run. |
| `GET /models` | Server-side model allowlist. |
| `POST /tasks`, `GET /tasks/{id}` | Seeded task sets (the public view never includes the optimum). |
| `POST /sessions`, `POST /sessions/{id}/step` | Interactive episode using the JSON protocol. |
| `POST /runs`, `GET /runs/{id}` | Bounded background runs. |
| `GET /leaderboard`, `GET /insights` | Leaderboard and plain-language analysis from saved results. |
| `GET /live?task_id=…&model=…` | Server-Sent Events stream of one episode. |

> [!WARNING]
> The server is a **single-user local demo**. It has no authentication, keeps sessions in memory and needs a single
> worker. Keys stay server-side: they are never logged, returned or saved, and clients can't supply URLs, paths or
> code. To share results publicly, use the static site (`sim2gym site`) instead of exposing the server.

### Static results site

```bash
.venv/bin/sim2gym site -o site
vercel deploy site --prod          # or any static host
```

The site bakes `/leaderboard`, `/insights` and the public task list into JSON. It includes complete runs only,
contains no optima and no keys, and hides live episodes.

### Programmatic and RL use

```python
from sim2gym.env import Sim2GymEnv
from sim2gym.tasks import load_tasks

env = Sim2GymEnv(load_tasks("tasks.jsonl"))
obs = env.reset(0)  # task prompt
obs, reward, done, info = env.step(
    '{"simulate": {"array_h": 16, "array_w": 64, "dataflow": "os", "sram_split": "balanced"}}'
)
# reward is 0 until the episode ends, then equals the task score in [0, 1]
```

### Task generator

`sim2gym generate` samples new layers from a seed and builds tasks with the same exhaustive optimum:

- **GEMMs** shaped like transformer projections, from single-token decode to short prefill: M ∈ {1 … 256}, K and
  N ∈ {64 … 768}, including non-powers of two.
- **Convolutions** on 7, 14 and 28-pixel inputs, with 1×1 or 3×3 filters, stride 1 or 2, and 16–512 channels.
- Layers are capped at 8M multiply-adds, the size of the largest fixed layer. The generator never repeats a fixed
  layer or a shape within a set, and the same seed always gives the same tasks.

### `verifiers` environment

`src/sim2gym_vf` exposes AccelGym as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 environment
(id `sim2gym-vf`) for open-source RL trainers and the Environments Hub. The reward is `sim2gym_score`, with
`solved`, `feasible` and `sim_calls_used` as metrics.

```bash
uv pip install -p .venv/bin/python -e '.[dev,vf]'
.venv/bin/vf-eval sim2gym-vf -m <model> --client.base-url <openai-compatible-url> \
  --client.api-key-var <ENV_VAR> --env.taskset.path tasks.jsonl \
  --env.player.runtime.type subprocess --no-push -n 36
.venv/bin/vf-eval sim2gym-vf ... --env.taskset.generate-seed 7 --env.taskset.generate-n 12   # fresh tasks
```

`--no-push` keeps results local; vf-eval uploads to Prime Intellect by default.

## Reproducibility

- **Deterministic tasks.** Task sets are identified by a content hash: fixed `903fb5045d2d4d26`, generated (seed 1)
  `2afed0ff8b2e201d`. `sim2gym build --seed 0` and `sim2gym generate --seed 1 --n 12` rebuild them byte-identically.
- **Cached simulator.** SCALE-Sim results are cached by input hash plus simulator version (which includes a hash of
  `packs/accel.py`). All 576 cached results for the fixed set were re-simulated from scratch with 0 differences.
- **Exact optimum.** Every task's optimum is the minimum over all feasible designs; the oracle scores exactly 1.000
  and no result anywhere scores above 1.0.
- **Exact baselines.** Random baselines are reported as exact expectations over every possible draw, computed
  combinatorially from the cached grid, not from one seeded draw.
- **Traceable numbers.** Every per-task record (model, settings, transcript, simulator log, grade) is in
  `results/*.jsonl`, and every number in this README is recomputed from those files.

## Project structure

```text
sim2gym/
├── src/sim2gym/
│   ├── packs/accel.py   # SCALE-Sim integration: design space, simulate(), cache
│   ├── tasks.py         # task building with exhaustive optima
│   ├── generate.py      # seeded procedural task generator
│   ├── protocol.py      # prompt rendering and JSON reply parsing
│   ├── grading.py       # scoring against the optimum
│   ├── env.py           # Sim2GymEnv: the single scoring path
│   ├── baselines.py     # oracle, 6-probe, random, max-square-ws, exact expectations
│   ├── models.py        # Bedrock, OpenRouter, OpenAI-compatible, Kiro and Codex backends
│   ├── runner.py        # parallel, resumable evaluation runs
│   ├── report.py        # leaderboards with CIs
│   ├── insights.py      # plain-language analysis for non-specialists
│   ├── site.py          # static results site
│   ├── api.py           # FastAPI server and SSE live episodes
│   ├── cli.py           # `sim2gym` command
│   └── static/          # single-page demo UI
├── src/sim2gym_vf/      # Prime Intellect verifiers adapter
├── tasks.jsonl          # fixed benchmark (36 tasks)
├── tasks_gen_s1.jsonl   # generated tasks, seed 1 (36 tasks)
├── leaderboards/       # generated Markdown leaderboards
├── results/, results_gen/  # per-task evaluation records
├── cache/               # SCALE-Sim result cache
└── tests/               # pytest suite
```

## Extending sim2gym

The environment needs only a plain simulation function. `src/sim2gym/packs/accel.py` is the whole SCALE-Sim
integration: it writes a config and topology file, runs `scalesim`, parses `COMPUTE_REPORT.csv` and caches the
result.

```python
def simulate(layer: dict, array_h: int, array_w: int, dataflow: str, sram_split: str) -> dict:
    ...  # run the real tool
    return {"cycles": ..., "stall_cycles": ..., "utilization_pct": ...}
```

To add another simulator, implement the same function plus `feasible_designs()` and `validate_design()`. Task
building, exact optima, grading, the environment, the runner, the API and the UI reuse them unchanged.

## Limitations

- One simulator so far, with a small discrete design space (4 × 4 × 3, SRAM split fixed) and small layers so that
  exhaustive search stays fast.
- SCALE-Sim models cycles, not area or power; the PE budget stands in for area. Reported cycles cover compute plus
  stalls, excluding the initial prefetch and the final DRAM drain.
- One run per task at temperature 0, 36 tasks per set: enough to see patterns, not to claim fine-grained rankings.
- Kiro and Codex results include each CLI's own system prompt. Each turn replays the conversation as text.
- GPT-6.1 Sol, Terra and Luna and Gemma 4 E4B have not yet been evaluated on the generated set.
- OpenRouter's free tier allows about 50 free-model requests per day, so the free-model runs in `results_eval/` are
  partial and not ranked
  ([`leaderboards/openrouter_free_partial.md`](leaderboards/openrouter_free_partial.md)).
- Related work: LUMINA (Mar 2026) and PerfReasoning (Sep 2026) also test LLMs on hardware trade-offs. sim2gym's
  contribution is a generator of black-box, budgeted environments from real simulators, not a first chip-reasoning
  benchmark.

## Roadmap

- [ ] Publish `sim2gym-vf` on the Prime Intellect Environments Hub.
- [ ] Evaluate every model on the generated set and on more seeds.
- [ ] Multiple runs per task and non-zero temperatures.
- [ ] A second simulator pack: ngspice + SKY130 inverter sizing.
- [ ] RL training runs against `Sim2GymEnv`.

## Development

```bash
uv pip install -p .venv/bin/python -e '.[dev,vf]'
.venv/bin/python -m pytest -q        # 155 tests; the verifiers tests are skipped without the `vf` extra
```

The tests are hermetic: they need no network, no API keys and no model accounts. The Kiro and Codex adapters are
tested against fake CLI scripts.

## Contributing

Contributions are welcome, especially new simulator packs, model backends and evaluation runs.

1. Fork the repository and create a branch (`git checkout -b feat/my-change`).
2. Make your change with tests, and run `pytest -q`.
3. If you add results, commit the `results/*.jsonl` records and regenerate the leaderboard with `sim2gym report`.
   Every reported number must be reproducible from committed files.
4. Open a pull request describing what changed and how you checked it.

Never commit API keys: `.env` is gitignored, and keys must only ever be read server-side.

## Acknowledgements

- [SCALE-Sim](https://github.com/scalesim-project/scale-sim-v2) (Georgia Tech), the cycle-level systolic-array
  simulator that grades every answer.
- [Prime Intellect verifiers](https://github.com/PrimeIntellect-ai/verifiers) for the RL environment interface.
- Model access through Amazon Bedrock, OpenRouter, Ollama, Kiro and Codex.
- Built at Hacktoberfest Hack Day × AI for Vizag (MLH), November 2026.

## Citation

```bibtex
@software{sim2gym2026,
  title  = {sim2gym: Real Simulators as Budgeted Black-Box RL Environments},
  author = {{sim2gym contributors}},
  year   = {2026},
  url    = {https://github.com/vijay-2155/sim2gym}
}
```

## License

Released under the [MIT License](LICENSE).
