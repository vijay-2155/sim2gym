# sim2gym — Project Context

> Turn any simulator into a benchmark and an RL environment for AI models.
> Flagship pack: **ChipGym**, five semiconductor simulators from transistor to AI accelerator.

This file is the single source of truth for the project. Give it to teammates and to coding agents
(rename it to `CLAUDE.md` if you want Claude Code to load it automatically).

---

## 1. Event context

| | |
|---|---|
| Event | Hacktoberfest Hack Day x AI for Vizag (MLH) |
| Date / place | Nov 1, 2026 · Ratan Tata Innovation Hub, 5th floor, The Deck Building, Siripuram, Visakhapatnam |
| Track | **Best Open-Source AI Project**: "an original project that uses open-source or open-weight AI as an important part of how it works." |
| Second track (optional) | Best Use of Gemma 4. Using Gemma 4 as a main evaluated model may qualify for both |
| Judging | usefulness, open-source AI usage, execution, demo quality, documentation |
| Submission | title, description, demo link or video, repository link, screenshots, tools used |
| Team | solo or up to 3, each member registered |
| Rules from organizers | Building beforehand is allowed. **Working product only, no slides-only demos.** Demos start **after lunch**, so the project must be demo-ready by lunch. Attendance is mandatory. A LinkedIn post about the selection is required (instructions to follow) |

---

## 2. One-line pitch

> AI chip companies make trade-offs every day: power vs speed, one big die vs chiplets, compute vs memory.
> Can AI models reason about them? **sim2gym turns chip simulators into benchmarks and RL environments,
> and shows exactly where open models fall short.**

---

## 3. The problem

- Training and evaluating AI reasoning needs **tasks with checkable answers**. Today that mostly means maths and code.
- Engineering and science are full of checkable problems: every field has simulators and formulas with exact answers.
- Turning one into a benchmark or training environment is slow custom work: writing questions, grading, units,
  finding the best possible answer, wiring it to training code. So it rarely happens.
- **sim2gym makes it one step:** wrap a Python function → get unlimited fresh tasks, code-based grading,
  baselines, a leaderboard and an RL environment.

### Who uses it

1. **RL researchers**: need environments with checkable rewards beyond maths and code.
2. **Eval builders**: need fresh tasks that models can't have memorized (new numbers every time).
3. **Domain experts** (chip architects, engineers): already have simulators and want to know if models can reason in their field.

### Prior art (position honestly)

- **Reasoning Gym** (open-thought, NeurIPS 2025 D&B): 100+ fixed procedural puzzle generators with checkable answers.
  sim2gym is complementary: *bring your own simulator*, plus design tasks and experiment budgets.
- **Chip AI benchmarks** such as VerilogEval and RTLLM mostly test writing Verilog (chip description code). sim2gym tests
  *quantitative design trade-offs*. (Verify with a quick search before pitching this claim.)
- **Industry already uses AI/RL on chip simulators**: Google AlphaChip (RL for TPU floorplanning), Cadence Cerebrus,
  Synopsys.ai, Nvidia ChipNeMo. This validates the direction; sim2gym is the simple open version for open models.

---

## 4. Core concepts

### 4.1 Simulator

A normal Python function that takes named inputs and returns a dict of named outputs, wrapped with a decorator
that declares input ranges, units and design goals.

```python
from sim2gym import simulator, Param, Choice, Design

@simulator(
    name="cmos_power",
    description=(
        "Digital CMOS block. Dynamic power P = alpha*C*V^2*f plus leakage V*I_leak. "
        "Max clock f_max = K*(V - V_T)^1.3 / V (alpha-power law)."
    ),
    params={
        "alpha":      Param(0.05, 0.3, unit=""),         # activity factor
        "C_nF":       Param(1, 50, unit="nF"),            # switched capacitance
        "V":          Param(0.6, 1.2, unit="V"),
        "f_GHz":      Param(0.5, 3.0, unit="GHz"),
        "V_T":        Param(0.30, 0.45, unit="V"),
        "K_GHz":      Param(3, 5, unit="GHz"),
        "I_leak_A":   Param(0.01, 0.5, unit="A"),
    },
    outputs={"power_W": "W", "fmax_GHz": "GHz", "slack_GHz": "GHz"},
    valid=lambda p: p["V"] > p["V_T"] + 0.1,
    designs=[Design(
        free=["V", "f_GHz"],
        require={"f_GHz_out": (">=", Param(1.0, 2.5, unit="GHz")), "slack_GHz": (">=", 0.0)},
        minimize="power_W",
    )],
)
def cmos_power(alpha, C_nF, V, f_GHz, V_T, K_GHz, I_leak_A):
    fmax = K_GHz * (V - V_T) ** 1.3 / V
    power = alpha * C_nF * V**2 * f_GHz + V * I_leak_A    # nF * GHz = 1, so the result is in W
    return {"power_W": power, "fmax_GHz": fmax, "slack_GHz": fmax - f_GHz, "f_GHz_out": f_GHz}
```

Building blocks:

- `Param(low, high, unit, integer=False, log=False)`: a numeric input range.
- `Choice([...])`: a categorical input (for example `Choice(["dry", "wet"])`).
- `valid=`: a rejection rule for physically invalid samples (for example a MOSFET must be in saturation).
- `Design(free=[...], require={...}, minimize=... | maximize=... | None)`:
  - `free`: the inputs the model chooses. All other inputs are sampled and fixed.
  - `require`: constraints on outputs, as `(op, value)` with op `>=`, `<=` or `between`. The value can be a `Param`, in which case the target is sampled per task.
  - `minimize` / `maximize`: an optional objective. With none, a design scores 1 if every constraint is met.

### 4.2 Task types

| Type | Question | Grading |
|---|---|---|
| **predict** | "Given these inputs, what is `<output>` in `<unit>`?" | Run the simulator. Score 1 if within relative tolerance (default 2%), else 0 |
| **whatif** | "If `<input>` is multiplied by k, does `<output>` increase, decrease or not change?" | Run the simulator twice. Exact label match → 1 |
| **design** | "Choose `<free inputs>` so that `<constraints>`, minimizing/maximizing `<objective>`." | Run the model's design. Infeasible or out of bounds → 0. Feasible → objective ratio vs. the optimizer's best, clipped to [0, 1]. No objective → 1 |

Design score:
- minimize: `score = optimum / model_value`
- maximize: `score = model_value / optimum`

### 4.3 Experiment budget (the RL part)

On design tasks the model may call the simulator up to `budget` times (default 5) before answering.
This turns the benchmark into a multi-step environment that measures **how efficiently a model experiments**.
Report the number of simulator calls used per task.

### 4.4 Agent protocol (plain JSON, works with any model)

Do not rely on native tool calling; it is unreliable on some open-model servers. The model ends every reply with
one JSON object on its last line:

```text
{"simulate": {"V": 0.85, "f_GHz": 2.0}}     # design tasks only, uses 1 budget unit
{"answer": 123.4}                            # predict
{"answer": "increase"}                       # whatif: increase | decrease | no_change
{"answer": {"V": 0.82, "f_GHz": 2.0}}        # design
```

- The environment replies to `simulate` with the outputs as JSON, or an error such as out of bounds or budget exhausted.
- Parse the **last** valid JSON object in the reply. Accept numbers in scientific notation.
- Invalid or missing JSON gets one retry message, then a score of 0.
- Max turns = `budget + 2`.

### 4.5 Reproducibility rules

- All sampling is seeded: the same seed gives the same tasks.
- Round sampled inputs to 3 significant figures **before** computing expected answers, so the prompt and the answer always match.
- Each task stores the simulator name and a version hash of its source code.
- Design optimum: integer inputs by brute force over the range; continuous inputs by `scipy.optimize.differential_evolution`
  (seeded) with constraint penalties, then a feasibility check. Store the optimal inputs and value in the task.
- What-if: choose the multiplier k from {0.5, 1.5, 2} so the new value stays inside the range. "no_change" uses
  relative tolerance 1e-9. Report class balance.

---

## 5. ChipGym pack (`sim2gym/packs/chips.py`)

Five simulators, one per level of the chip stack. All are **first-order textbook models**: real models engineers use
for early decisions, but simplified. Each prompt states the model and its constants so every question has exactly one
correct answer. Parameter values are stylized, not real product numbers.

### 5.1 Transistor: `mosfet_saturation` (long-channel square law)

- `I_D = 0.5 * muCox * (W/L) * (V_GS - V_T)^2 * (1 + lambda * V_DS)`, with muCox in µA/V², so I_D is in µA.
- Valid only in saturation: `V_GS > V_T` and `V_DS >= V_GS - V_T` (use `valid=`).
- Params: `muCox_uA_V2` 100–400, `W_um` 0.5–50, `L_um` 0.18–2, `V_GS` 0.6–1.8, `V_T` 0.3–0.6, `V_DS` 0.5–1.8, `lambda_per_V` 0.02–0.1.
- Outputs: `I_D_uA`, `area_um2 = W*L`.
- Design: free `W_um`, `L_um`; require `I_D_uA >= target` (sampled 50–2000); minimize `area_um2`.
  Insight: the best design uses minimum L. Check whether models realize this.
- Check: muCox=200, W/L=10, V_ov=0.5, lambda=0 → **250 µA**.

### 5.2 Manufacturing: `oxide_growth` (Deal–Grove)

- `x^2 + A x = B (t + tau)`, solved as `x = (A/2) * (sqrt(1 + (t + tau) / (A^2 / (4B))) - 1)`.
- Arrhenius: `B = C1 * exp(-E1/kT)`, `B/A = C2 * exp(-E2/kT)`, k = 8.617e-5 eV/K, T in kelvin.
- Constants (Jaeger, *Introduction to Microelectronic Fabrication*; **double-check against the source**):
  - dry O2: C1 = 7.72e2 µm²/h, E1 = 1.23 eV; C2 = 6.23e6 µm/h, E2 = 2.0 eV
  - wet (H2O): C1 = 3.86e2 µm²/h, E1 = 0.78 eV; C2 = 1.63e8 µm/h, E2 = 2.05 eV
- `tau = (x_i^2 + A x_i) / B` for initial oxide x_i.
- Params: `temp_C` 800–1200, `time_h` 0.1–10, `ambient` Choice(dry, wet), `x_i_nm` 0–25.
- Output: `thickness_nm`.
- Design: free `time_h`; require `thickness_nm` between target ±2% (target sampled 20–500 nm); no objective (inverse problem).
- Check: dry, 1000 °C, 1 h, x_i=0 → **about 54 nm** with these constants (textbook charts show roughly 50 nm).

### 5.3 Circuit: `cmos_power` (power vs speed, DVFS)

- See the example in 4.1: `P = alpha*C*V^2*f + V*I_leak`, `f_max = K*(V - V_T)^1.3 / V`.
- Design: free `V`, `f_GHz`; require clock ≥ target and slack ≥ 0; minimize power.
  Optimum: the lowest voltage that still meets the target clock.
- Check: alpha=0.1, C=10 nF, V=1.0, f=2 GHz, I_leak=0 → **2.0 W**.

### 5.4 Factory economics: `chiplet_cost` (headline task 1)

- Dies per wafer: `DPW = floor(pi*(d/2)^2 / A - pi*d / sqrt(2A))`, with d = wafer diameter (mm) and A = die area (mm²).
- Yield (negative binomial): `Y = (1 + A_cm2 * D0 / a)^(-a)`, with D0 in defects/cm² and a = clustering factor.
- Split the total logic area into n equal chiplets: `A_chiplet = A_total/n + (d2d_mm2 if n > 1 else 0)` (die-to-die interface overhead).
- `cost_good_die = wafer_cost / (DPW * Y)`.
- Package: `pkg = package_base + bond_cost * n` (n=1 → bond_cost 0); assembly yield `bond_yield^n` (n=1 → 1).
- `cost_per_good_chip = (n * cost_good_die + pkg) / assembly_yield`.
- Reticle limit: `A_chiplet <= 858 mm²` (26 × 33 mm), otherwise the chip can't be printed.
- Params: `total_area_mm2` 100–1200, `D0_per_cm2` 0.05–0.3, `cluster_a` 2–4, `wafer_cost_usd` 5,000–20,000,
  `wafer_d_mm` 300, `d2d_mm2` 2–15, `package_base_usd` 20–200, `bond_cost_usd` 5–50, `bond_yield` 0.97–0.995,
  `n_chiplets` integer 1–8.
- Outputs: `cost_per_good_chip_usd`, `die_yield`, `dies_per_wafer`, `chiplet_area_mm2`.
- Design: free `n_chiplets`; require `chiplet_area_mm2 <= 858`; minimize `cost_per_good_chip_usd`. Brute force gives the optimum.
- Checks: 300 mm wafer, 100 mm² → **DPW = 640**. A = 1 cm², D0 = 0.1, a = 3 → **Y ≈ 0.906**.

### 5.5 AI accelerator: `llm_accelerator` (roofline, headline task 2)

- Chip area budget split between compute and memory interfaces:
  - `peak_TFLOPs = compute_area_mm2 * tflops_per_mm2`
  - `BW_TBps = n_hbm * bw_per_stack_TBps`
  - `area_used = compute_area_mm2 + n_hbm * phy_area_mm2`
- LLM decode step with batch B, N billion params, bytes per param b:
  - FLOPs per step = `2 * N * 1e9 * B`
  - bytes per step = `N * 1e9 * b` (weights read once per step; KV cache and attention ignored, stated in the prompt)
  - `step_time = max(flops / (peak_TFLOPs*1e12), bytes / (BW_TBps*1e12))`
  - `tokens_per_s = B / step_time`; `bound` = "memory" or "compute".
- Params: `area_budget_mm2` 300–800, `tflops_per_mm2` 0.5–3, `bw_per_stack_TBps` 0.8–1.6, `phy_area_mm2` 10–30,
  `model_params_B` 7–70, `bytes_per_param` Choice(1, 2), `batch` integer 1–64, `compute_area_mm2` free, `n_hbm` integer 1–8.
- Design: free `compute_area_mm2`, `n_hbm`; require `area_used <= area_budget_mm2`; maximize `tokens_per_s`.
  Optimum: enumerate n_hbm, give the rest of the area to compute.
- Insight for the demo: decode at small batch is **memory-bound**. A model that spends its area on compute builds a slow chip.
- Check: 70B params, FP16, batch 1, 3 TB/s → **≈ 21.4 tokens/s** (memory-bound).

### 5.6 Real-tool layer (stretch, only if installs work)

1. **ngspice + SkyWater SKY130 open PDK**: CMOS inverter sizing. Free `Wp`, `Wn`; require switching threshold V_M within
   VDD/2 ± 2%; minimize area. Graded by a real SPICE DC sweep using real factory transistor models.
   This is the strongest "not a toy" proof. Risk: Windows install and large PDK files. Test before promising it.
2. **SCALE-Sim** (systolic-array AI accelerator simulator): choose array rows and columns for a given GEMM under a PE budget;
   minimize cycles. Check its install on Windows first.

If neither installs cleanly, present them as next steps. The core project does not depend on them.

### 5.7 Generic examples (`packs/examples.py`)

One or two tiny non-chip simulators (for example a loan EMI or a projectile range) that prove the framework works for
anything. One of them is the backup for the live "a judge names a formula" demo.

---

## 6. Architecture

```text
sim2gym/
  pyproject.toml            # name: sim2gym, python >=3.10, deps: numpy, scipy, openai; dev: pytest
  LICENSE                   # MIT
  README.md
  PROJECT_CONTEXT.md        # this file
  src/sim2gym/
    __init__.py             # public API: simulator, Param, Choice, Design, get_sim, list_sims
    core.py                 # dataclasses, Simulator wrapper, registry, version hash
    sampling.py             # seeded sampling, 3-sig-fig rounding, valid= rejection
    tasks.py                # Task schema + generators: predict / whatif / design
    optimize.py             # optimum search (brute force for ints, differential_evolution for floats)
    grading.py              # score functions
    protocol.py             # prompt templates + last-JSON parsing
    env.py                  # Sim2GymEnv: reset() / step(text) -> (obs, reward, done, info)
    agents.py               # oracle (uses the simulator, upper bound) and random (lower bound)
    runner.py               # OpenAI-compatible chat loop, thread pool, results JSONL, resume
    report.py               # leaderboard: model x task type, per-simulator breakdown, markdown
    cli.py                  # sim2gym list | build | run | report
    packs/
      chips.py              # ChipGym: the 5 simulators above
      examples.py           # tiny generic simulators
  tests/
    test_chip_physics.py    # the textbook checks listed in section 5
    test_grading.py         # correct → 1, wrong → 0, tolerance edges, infeasible design → 0
    test_oracle.py          # oracle scores 1.0 on every generated task
    test_tasks.py           # seeding reproducible, rounding consistent, JSONL round trip
    test_protocol.py        # JSON parsing edge cases
  examples/
    my_simulator.py         # "add your own simulator in 10 lines"
```

### CLI

```bash
sim2gym list                                        # registered simulators
sim2gym build chips --n 300 --seed 0 -o tasks.jsonl
sim2gym run tasks.jsonl --model oracle               # sanity check, must score 1.0
sim2gym run tasks.jsonl --model random
sim2gym run tasks.jsonl --model gemma4:e4b --base-url http://localhost:11434/v1 --workers 4
sim2gym report results/ -o leaderboard.md
```

- Endpoints: any OpenAI-compatible server. Ollama `http://localhost:11434/v1`; Gemini API
  `https://generativelanguage.googleapis.com/v1beta/openai/` (key from env `SIM2GYM_API_KEY`). Never hard-code keys.
- Use `--module path/to/file.py` to load user-defined simulators.

### RL environment API

```python
from sim2gym import load_tasks
from sim2gym.env import Sim2GymEnv

env = Sim2GymEnv(load_tasks("tasks.jsonl"), budget=5)
obs = env.reset(0)                       # prompt text for task 0
obs, reward, done, info = env.step(model_reply_text)
# reward is 0 until done, then the task score in [0, 1]
```

The runner is just a loop that drives `Sim2GymEnv` with a model, so evaluation and RL use the same code path.

### Task JSONL schema (one line per task)

```json
{
  "id": "chiplet_cost/design/000042",
  "sim": "chiplet_cost",
  "sim_version": "a1b2c3d4",
  "kind": "design",
  "seed": 0,
  "params": {"total_area_mm2": 820, "D0_per_cm2": 0.12, "...": "..."},
  "free": ["n_chiplets"],
  "require": {"chiplet_area_mm2": ["<=", 858]},
  "objective": {"minimize": "cost_per_good_chip_usd"},
  "optimum": {"inputs": {"n_chiplets": 3}, "value": 412.7},
  "prompt": "...",
  "budget": 5
}
```

Predict tasks store `target_output`, `expected` and `rel_tol`. What-if tasks store `changed_param`, `factor`,
`target_output` and `expected_label`. The file loads as a Hugging Face dataset with `load_dataset("json", ...)`.

### Results JSONL (one line per task × model)

`task_id, model, kind, sim, score, feasible, answer, sim_calls_used, turns, latency_s, error, transcript`

### Report metrics

- Per model × task type: n, mean score, pass rate (score = 1) for predict and whatif.
- Design: feasible rate, mean % of optimal, mean simulator calls used.
- Per-simulator breakdown, plus oracle and random rows as bounds.
- Output: a markdown table for the README, optionally a small Streamlit page for the demo.

---

## 7. Definition of done ("not a toy" checklist)

P0, must have before the demo:
- [ ] `pip install -e .` works; `sim2gym list | build | run | report` work end to end.
- [ ] All 5 chip simulators implemented with the section 5 checks passing as tests.
- [ ] Oracle scores 1.0 on every task; random scores near 0; wrong answers score 0 (tests).
- [ ] Seeded and reproducible; prompts and expected answers consistent after rounding.
- [ ] Runs against at least 2 open models (Gemma 4 + one other) with a real leaderboard.
- [ ] README: pitch, quickstart, "add your own simulator in 10 lines", leaderboard, limitations.

P1, should have:
- [ ] Experiment budget on design tasks with `simulate` calls counted.
- [ ] One generic example simulator added live in under 2 minutes (rehearsed).
- [ ] Demo view: leaderboard plus one design failure shown as simulator output.

P2, stretch:
- [ ] ngspice + SKY130 inverter task, or SCALE-Sim task.
- [ ] Gymnasium-style adapter / TRL or verifiers integration example.
- [ ] Publish the task set to the Hugging Face Hub.

---

## 8. Demo script (5 minutes)

1. **Problem (30 s):** "AI chip companies make trade-offs daily. Can AI models reason about them? There's no easy way to test or train that."
2. **Framework (60 s):** show a ~10-line simulator → `sim2gym build chips` → 300 fresh, auto-graded tasks.
3. **Results (90 s):** leaderboard of Gemma 4 vs another open model across predict / whatif / design, with oracle and random as bounds.
4. **The finding (60 s):** one failed design shown as simulator output, for example a model buying compute when LLM decode is memory-bound.
   Use whatever the real results show; don't invent a finding.
5. **Live (60 s):** a judge names a formula → add it → run a brand-new benchmark. Keep the backup example ready.

Close: "Open-source, pip-installable, works with any simulator and any open model, ready for RL training."

---

## 9. Honest limits (say these in the demo and README)

- v0.1: a solid, tested core, not a mature framework.
- ChipGym uses first-order textbook models, not production EDA tools (SPICE/TCAD). They are real early-design models,
  but simplified. The ngspice layer is the path to industry-grade grading.
- No RL training run yet: it needs a GPU (AWS GPU quota is currently 0; request it for after the event).
- Simple formula tasks may be easy for models; the design tasks with experiment budgets are the core value.

---

## 10. Engineering conventions

- Python ≥ 3.10, type hints, dataclasses; minimal dependencies (numpy, scipy, openai). No heavy frameworks in the core.
- Units are always explicit in names (`_uA`, `_mm2`, `_GHz`) and in prompts.
- Every simulator: pure function, deterministic, no I/O, returns a dict of floats (and strings for labels like `bound`).
- Tests before features: if grading is wrong, the benchmark is worthless.
- Secrets only from environment variables. Never commit API keys.
- Temperature 0 by default for evaluations; record model name, endpoint and settings in results.

---

## 11. Submission checklist

- [ ] Title: "sim2gym: turn any simulator into a benchmark & RL environment (ChipGym)"
- [ ] Description (from sections 2–3)
- [ ] Demo link or video (record a 2-minute backup video in case venue Wi-Fi fails)
- [ ] Public repository link with README and leaderboard
- [ ] Screenshots: leaderboard, a design failure, the 10-line simulator
- [ ] Tools used: Python, NumPy, SciPy, Gemma 4 (Ollama / Gemini API), other open models, Claude Code
- [ ] Track: Best Open-Source AI Project (mention Gemma 4 usage)
- [ ] LinkedIn post per the organizers' instructions
