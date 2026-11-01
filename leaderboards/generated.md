# sim2gym leaderboard (AccelGym on SCALE-Sim)

Task set `2afed0ff8b2e201d`

| model | type | n | mean score | 95% CI | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | incomplete (infra errors) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | baseline | 36 | 1.000 | - | 100% | 100% | 0.00 | 0.00 | 0 |
| kiro:claude-opus-5.5 | model | 36 | 0.942 | +/-0.050 | 83% | 100% | 3.78 | 108.52 | 0 |
| bedrock:nvidia.nemotron-super-3-120b | model | 36 | 0.910 | +/-0.052 | 61% | 100% | 5.00 | 6.00 | 0 |
| bedrock:deepseek.v3.2 | model | 36 | 0.866 | +/-0.060 | 50% | 100% | 5.17 | 62.97 | 0 |
| bedrock:google.gemma-3-27b-it | model | 36 | 0.816 | +/-0.073 | 42% | 100% | 4.78 | 31.17 | 0 |
| bedrock:qwen.qwen3-32b-v1:0 | model | 36 | 0.803 | +/-0.101 | 53% | 92% | 4.97 | 7.19 | 0 |
| 6-probe (exact expectation) | baseline | 36 | 0.789 | exact | 36% | 100% | 6.00 | 0.00 | 0 |
| bedrock:openai.gpt-oss-120b-1:0 | model | 36 | 0.704 | +/-0.096 | 31% | 97% | 2.81 | 11.04 | 0 |
| bedrock:mistral.mistral-large-3-675b-instruct | model | 36 | 0.549 | +/-0.112 | 25% | 92% | 0.78 | 4.77 | 0 |
| max-square-ws | baseline | 36 | 0.464 | +/-0.086 | 6% | 100% | 0.00 | 0.00 | 0 |
| random (exact expectation) | baseline | 36 | 0.444 | exact | 7% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. n = tasks with a graded answer; rate-limit/provider failures are excluded and shown as incomplete. Baselines are not model inference. 95% CI = +/-1.96 standard errors over tasks (one run per task). Exact-expectation rows average a random baseline over every possible draw; the plain random and 6-probe rows are one seeded draw.
