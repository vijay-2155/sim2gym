# sim2gym leaderboard (AccelGym on SCALE-Sim)

> WARNING: task set 6f73cea2f0592ee1: models were run on different numbers of tasks [1, 2, 3, 12]

Task set `6f73cea2f0592ee1`

| model | type | n | mean score | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | incomplete (infra errors) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openrouter:nvidia/nemotron-3-ultra-550b-a55b:free | model | 1 | 1.000 | 100% | 100% | 5.00 | 63.55 | 1 |
| oracle | baseline | 12 | 1.000 | 100% | 100% | 0.00 | 0.00 | 0 |
| openrouter:dots-studio/dots-3-note-preview:free | model | 2 | 0.839 | 50% | 100% | 5.50 | 70.19 | 1 |
| bedrock:qwen.qwen3-32b-v1:0 | model | 12 | 0.799 | 50% | 100% | 5.00 | 8.55 | 0 |
| bedrock:nvidia.nemotron-super-3-120b | model | 12 | 0.799 | 50% | 100% | 5.00 | 6.41 | 0 |
| bedrock:deepseek.v3.2 | model | 12 | 0.785 | 50% | 100% | 5.17 | 63.40 | 0 |
| openrouter:nvidia/nemotron-3.5-super-vl:free | model | 3 | 0.773 | 33% | 100% | 5.33 | 83.02 | 1 |
| bedrock:google.gemma-3-27b-it | model | 12 | 0.704 | 25% | 100% | 5.17 | 33.31 | 0 |
| 6-probe | baseline | 12 | 0.702 | 25% | 100% | 6.00 | 0.00 | 0 |
| bedrock:openai.gpt-oss-120b-1:0 | model | 12 | 0.568 | 17% | 100% | 2.00 | 9.37 | 0 |
| bedrock:mistral.mistral-large-3-675b-instruct | model | 12 | 0.540 | 17% | 100% | 1.25 | 6.40 | 0 |
| max-square-ws | baseline | 12 | 0.524 | 0% | 100% | 0.00 | 0.00 | 0 |
| random | baseline | 12 | 0.458 | 8% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. n = tasks with a graded answer; rate-limit/provider failures are excluded and shown as incomplete. Baselines are not model inference.
