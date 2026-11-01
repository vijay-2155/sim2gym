# sim2gym leaderboard (AccelGym on SCALE-Sim)

Task set `903fb5045d2d4d26`

| model | type | n | mean score | 95% CI | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | incomplete (infra errors) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | baseline | 36 | 1.000 | - | 100% | 100% | 0.00 | 0.00 | 0 |
| codex:gpt-6.1-sol | model | 36 | 0.861 | +/-0.067 | 64% | 100% | 5.67 | 67.53 | 0 |
| codex:gpt-6.1-terra | model | 36 | 0.838 | +/-0.079 | 61% | 97% | 5.61 | 52.68 | 0 |
| bedrock:nvidia.nemotron-super-3-120b | model | 36 | 0.838 | +/-0.071 | 56% | 100% | 5.00 | 5.34 | 0 |
| kiro:claude-opus-5.5 | model | 36 | 0.825 | +/-0.070 | 56% | 100% | 4.17 | 120.03 | 0 |
| bedrock:qwen.qwen3-32b-v1:0 | model | 36 | 0.804 | +/-0.084 | 53% | 100% | 5.06 | 7.41 | 0 |
| openai:gemma4:e4b | model | 36 | 0.726 | +/-0.085 | 33% | 97% | 5.67 | 46.72 | 0 |
| 6-probe (exact expectation) | baseline | 36 | 0.723 | exact | 31% | 100% | 6.00 | 0.00 | 0 |
| bedrock:google.gemma-3-27b-it | model | 36 | 0.709 | +/-0.094 | 36% | 100% | 4.97 | 33.97 | 0 |
| bedrock:openai.gpt-oss-120b-1:0 | model | 36 | 0.706 | +/-0.085 | 36% | 100% | 3.11 | 11.17 | 0 |
| bedrock:deepseek.v3.2 | model | 36 | 0.704 | +/-0.091 | 36% | 100% | 5.11 | 68.92 | 0 |
| 6-probe | baseline | 36 | 0.650 | +/-0.092 | 28% | 100% | 6.00 | 0.00 | 0 |
| codex:gpt-6-luna | model | 36 | 0.618 | +/-0.112 | 36% | 97% | 1.94 | 29.94 | 0 |
| bedrock:mistral.mistral-large-3-675b-instruct | model | 36 | 0.590 | +/-0.102 | 22% | 94% | 0.42 | 5.40 | 0 |
| max-square-ws | baseline | 36 | 0.521 | +/-0.086 | 8% | 100% | 0.00 | 0.00 | 0 |
| random (exact expectation) | baseline | 36 | 0.399 | exact | 6% | 100% | 0.00 | 0.00 | 0 |
| random | baseline | 36 | 0.380 | +/-0.094 | 8% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. n = tasks with a graded answer; rate-limit/provider failures are excluded and shown as incomplete. Baselines are not model inference. 95% CI = +/-1.96 standard errors over tasks (one run per task). Exact-expectation rows average a random baseline over every possible draw; the plain random and 6-probe rows are one seeded draw.
