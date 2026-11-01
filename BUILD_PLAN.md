# sim2gym: build plan for today (1 Nov 2026)

Hand this file to Claude Code. It replaces the "five textbook simulators" plan with one that grades models
using a **real open-source simulator**, so the demo is not a toy.

Read these first, in order:
1. This file. Where it disagrees with the files below, this file wins.
2. `PROJECT_CONTEXT.md`. Sections 4.2–4.5 (task types, budget, JSON protocol, reproducibility) and 10 (conventions) still apply. Section 5 (the five textbook simulators) is **superseded**.
3. `docs/research/sim2gym-market-viability-audit.md`. This is background only: why the textbook version was judged a toy.

---

## 1. What we are building

**sim2gym** turns real simulators into black-box RL environments and benchmarks for open models. It is built on top of three open-source projects:

| Open-source project | Role |
|---|---|
| **SCALE-Sim** (`scalesim` on PyPI; Georgia Tech systolic-array accelerator simulator, used in research) | The real simulator that grades every answer |
| **Prime Intellect `verifiers`** (P1 below) | Standard environment format, so the env plugs into existing RL trainers and the Environments Hub |
| **Ollama + open-weight models** (Gemma, Qwen) | The models being evaluated |

**Headline environment: AccelGym.** The model designs an AI-accelerator systolic array for a real neural-network layer:
- **Workloads:** real layer shapes from transformer and CNN models.
- **Choices:** array height and width, dataflow, and SRAM split.
- **Constraint:** a processing-element (PE) budget.
- **Goal:** minimize cycles.

The model never sees a formula. It can call the real SCALE-Sim up to a fixed budget of times before answering, and its answer is graded by running SCALE-Sim. This is the black-box, experiment-budget format the audit said makes it real.

**Pitch:**

> "Chip companies make accelerator trade-offs every day. sim2gym wraps a real open-source accelerator simulator as an RL environment: the model has to experiment, not recall a formula. Here's how open models do, and where they fail."

---

## 2. Machine facts (checked at 11:15 AM)

- Windows 11. Claude Code has both PowerShell and Git Bash available.
- **Python is not on PATH.** Use `C:\Users\Mani Varshith\anaconda3\python.exe` to create a venv at `sim2gym\.venv`, then always call `.venv\Scripts\python.exe`. Never call bare `python`.
- **Ollama is installed and running** at `http://localhost:11434`. Only `phi3:latest` (3.8B) is pulled so far.
  - `ollama list` hung when called from Git Bash. Use the HTTP API (`/api/tags`) or PowerShell instead.
- **GPU:** RTX 3050 Laptop, 6 GB VRAM. 4B models at Q4 fit, one at a time.
- **Not installed:** ngspice, conda on PATH, uv.
- **Not yet verified:** whether `scalesim` and `verifiers` install and run on this Windows machine. Step 1 checks this first.

---

## 3. Timeline (demo-ready by about 1:30 PM; adjust to the real lunch time)

Each block has a cut line. If a block runs over, take its fallback and move on. **The core loop must work by 1:15 PM:** real simulator → task → model → grade → leaderboard.

### Step 0 — Setup (11:30–11:45)
1. `git init` inside `sim2gym/`. Add a `.gitignore` covering `.venv/`, `results/`, `cache/`, `*.csv` outputs and `.env`.
2. Create the venv with the Anaconda Python. Install `numpy`, `scipy`, `openai`, `pytest` and `scalesim`.
3. **Start model downloads in the background now**, because they are the slowest step on venue Wi-Fi:
   - `ollama pull qwen3:4b`
   - a Gemma model: try `gemma4:e4b` first (Gemma 4 track), and fall back to `gemma3:4b` if it doesn't exist.
4. `pyproject.toml`: name `sim2gym`, Python ≥3.10, `src/` layout as in PROJECT_CONTEXT §6, MIT licence. **Before choosing the licence, ask the human whether the ownership question is settled** (section 7).

### Step 1 — Get SCALE-Sim running (11:45–12:05)
1. Inspect the installed `scalesim` package (its README, examples and `scale_sim.py`) and run one simulation **from Python**. The likely API is the following, but **verify it**; don't trust this signature:
   ```python
   from scalesim.scale_sim import scalesim
   s = scalesim(save_disk_space=True, verbose=False, config=cfg_path, topology=topo_path)
   s.run_scale(top_path=out_dir)
   ```
   - It reads a `.cfg` file (`ArrayHeight`, `ArrayWidth`, `IfmapSramSzkB`, `FilterSramSzkB`, `OfmapSramSzkB`, `Dataflow` = os|ws|is, `Bandwidth`) and a topology CSV of layers.
   - It writes `COMPUTE_REPORT.csv`, which has total cycles, stall cycles and utilization.
2. Write `src/sim2gym/packs/accel.py` with a pure function:
   ```python
   simulate(layer: dict, array_h: int, array_w: int, dataflow: str, sram_split: str) -> dict
   ```
   - It writes temp config and topology files, runs SCALE-Sim, parses the report and returns `{"cycles", "utilization_pct", "stall_cycles"}`.
   - **Cache** every result to disk under `cache/`, keyed by a hash of the inputs plus the scalesim version.
3. **Measure the runtime** of one call for a small and a medium layer. This number sets the grid size in step 2.

**Fallbacks if `scalesim` fails on Windows** (time box: 20 minutes):
1. `pip install git+https://github.com/scalesim-project/scale-sim-v2`.
2. Check whether WSL is available (`wsl -l -v`) and run SCALE-Sim there through a subprocess.
3. **Last resort:** use the analytical roofline `llm_accelerator` model from PROJECT_CONTEXT §5.5, labeled honestly as "first-order model, real-simulator support in progress". This is the toy path; tell the human before taking it.

### Step 2 — Tasks, optimum and grading (12:05–12:35)

**Design space** (discrete, so the true optimum can be found exhaustively):
- `array_h`, `array_w` ∈ {4, 8, 16, 32, 64, 128}, with `array_h * array_w ≤ PE_budget`. PE_budget is sampled per task from {256, 1024, 4096}.
- `dataflow` ∈ {os, ws, is}.
- `sram_split`: 3 presets of a fixed total SRAM (balanced, ifmap-heavy, filter-heavy).
- **If step 1's runtime is high,** drop `sram_split` (fix it at balanced) and shrink the shape set. Size the grid so the whole precompute takes under 10 minutes using `multiprocessing`.

**Workloads:**
- 10–16 real layer shapes: a few GEMMs from a small transformer (attention projection, MLP up and down, at small sequence lengths) and a few conv layers from ResNet-18 or MobileNet.
- Include **tall-skinny and wide shapes on purpose**, so the best array is often not square. This is what makes the task non-trivial.
- Keep dimensions small enough that one simulation takes seconds.

**Tasks:**
- One design task per (layer × PE budget) combination. Store everything in `tasks.jsonl` using the schema in PROJECT_CONTEXT §6, with `optimum.inputs` and `optimum.value` precomputed by exhaustive search over the cached grid.

**Grading:**
- Out of bounds or over the PE budget → 0.
- Otherwise `score = optimum_cycles / answer_cycles`, clipped to [0, 1].
- `solved` = score ≥ 0.95.
- Since the search is exhaustive, the optimum is exact; this closes the audit's "optimizer not proven optimal" hole.

**Tests** (`tests/`):
- The oracle scores 1.0 on every task.
- A wrong or infeasible answer scores 0.
- Seeding is reproducible.
- The cache returns results identical to a fresh run.
- The JSON parser handles edge cases.

### Step 3 — Multi-turn env and runner (12:35–1:05)
1. `env.py` implements `Sim2GymEnv` (`reset` / `step`) with the JSON protocol from PROJECT_CONTEXT §4.4:
   - `{"simulate": {...}}` runs SCALE-Sim (from the cache when possible), returns the metrics as JSON and uses one budget unit.
   - `{"answer": {...}}` ends the episode.
   - The budget is **6 simulator calls**, and max turns = budget + 2.
2. **The prompt** describes:
   - the workload layer;
   - the design-space choices and their allowed values;
   - the PE budget;
   - the objective (minimize cycles);
   - the protocol.

   **It contains no formulas and no hints** about which shape is best.
3. `runner.py`:
   - Uses an OpenAI-compatible client against Ollama (`http://localhost:11434/v1`, key `"ollama"`) with temperature 0.
   - Writes `results/<model>.jsonl`.
   - Can resume after an interruption.
   - Runs models one at a time; there's only 6 GB of VRAM.
4. **Baselines,** which don't need an LLM:
   - `oracle` (the stored optimum);
   - `random` (a random feasible design);
   - `max-square-ws` (the largest square array that fits, with the ws dataflow);
   - `6-probe` (6 random simulator calls, keeping the best feasible design).

   The 6-probe baseline is the important one: a model that doesn't beat it isn't reasoning.

### Step 4 — Results and report (1:05–1:20)
1. Run on all tasks: the baselines, `qwen3:4b`, the Gemma model, and `phi3` (already pulled).
2. `report.py` writes `leaderboard.md` with these columns per row:
   - mean score;
   - solve rate;
   - feasible rate;
   - mean simulator calls used;
   - mean latency.
3. Find **one real failure** for the demo. For example: a model that picks the largest square array for a tall-skinny GEMM, shown next to SCALE-Sim's cycles for its design vs. the optimum. **Use what the results actually show; don't invent a finding.**

### Step 5 — README and demo prep (1:20–1:30)
The README needs:
- the pitch;
- "built on SCALE-Sim, Ollama and (if P1 is done) `verifiers`";
- a quickstart;
- the leaderboard;
- one failure example;
- a "how to wrap your own simulator" section, showing that `accel.py` is a plain function;
- honest limits (section 6).

Also take screenshots of the leaderboard, the failure, and the simulator wrapper code.

---

## 4. Priorities and cut lines

| Priority | Item | Cut if behind schedule? |
|---|---|---|
| **P0** | SCALE-Sim wrapper + cache, design tasks with an exact optimum, grading, tests, multi-turn env with budget, runner on 2 open models + baselines, leaderboard, README | Never: this is the demo |
| **P1** | `verifiers` adapter: a `load_environment()` returning a multi-turn env wrapping `Sim2GymEnv`, checked with `vf-eval` against Ollama. Read the installed `verifiers` version's docs and examples first, because its API changed in 2026 (v1 Taskset/Harness). **Time box: 25 minutes after step 4.** If it fails on Windows, list it as next step | Yes |
| **P1** | Predict and what-if tasks on the same simulator (e.g., "which of these two configs has fewer cycles?"). Report what-if against the 50% guessing floor for a two-way choice | Yes |
| **P2** | A second real tool: ngspice + SKY130 inverter sizing. Don't attempt it today: the install on Windows and the PDK download are too risky | Yes: say "next" in the demo |
| **P2** | The textbook ChipGym simulators as a "sanity tier" | Yes |

---

## 5. Demo script (5 minutes)

1. **Problem (30 s):** accelerator design is a trade-off. Can open models reason about it, or do they just pattern-match?
2. **Built on open source (60 s):** show `accel.py`: a plain function wrapping the real SCALE-Sim. Show `sim2gym build`, which produces tasks with exact optima.
3. **Live (60 s):** one episode in which a model calls the simulator, sees the cycles and answers.
4. **Results (90 s):** the leaderboard: open models vs. the 6-probe baseline vs. the oracle.
5. **The finding, then next steps (60 s):**
   - the real failure from step 4;
   - next: `verifiers` / Environments Hub (if not done), ngspice + SKY130, and RL training.

---

## 6. Honest limits (say these in the README and the demo)

- v0.1, built in one day.
- One real simulator so far, with a small discrete design space and small layers to keep simulation fast.
- Evaluated on small local open models only (6 GB GPU); no RL training run yet.
- SCALE-Sim models cycles, not area or power. A PE budget stands in for area.
- The novelty claim must be accurate. LUMINA (Mar 2026) and PerfReasoning (Sep 2026) already test LLMs on hardware trade-offs. sim2gym's contribution is a **generator of black-box, budgeted environments from real simulators**, not "the first chip-reasoning benchmark".

---

## 7. Human tasks (not for Claude Code)

- **Ownership:** message the New Measure founders before the repo goes public (audit, risk 1). If there's no answer by submission time, ask the organizers whether judges can access a private repo.
- **Name check:** search GitHub and PyPI for "sim2gym" and "AccelGym".
- **Optional:** a Gemini API key for hosted Gemma, set as the env var `SIM2GYM_API_KEY` and never committed.
- Record a 2-minute backup video once step 4 works.
- Submission: title, description, demo video, repo link, screenshots, tools used (SCALE-Sim, Ollama, Gemma, Qwen, `verifiers`, Python, Claude Code), and the track.
- The LinkedIn post the organizers require.

## 8. Rules for Claude Code

- Windows: use PowerShell or `.venv\Scripts\python.exe` explicitly. Run long jobs (model pulls, precompute, eval runs) in the background.
- Tests come before features. If grading is wrong, the benchmark is worthless.
- Never invent results or findings. Every number in the README comes from `results/`.
- No API keys in code or commits.
- Commit after each step that passes its tests.
