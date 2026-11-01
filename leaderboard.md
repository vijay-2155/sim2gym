# sim2gym leaderboard (AccelGym on SCALE-Sim)

> WARNING: task set 903fb5045d2d4d26: models were run on different numbers of tasks [22, 36]

Task set `903fb5045d2d4d26`

| model | type | n | mean score | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | incomplete (infra errors) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | baseline | 36 | 1.000 | 100% | 100% | 0.00 | 0.00 | 0 |
| bedrock:nvidia.nemotron-super-3-120b | model | 36 | 0.838 | 56% | 100% | 5.00 | 5.34 | 0 |
| bedrock:qwen.qwen3-32b-v1:0 | model | 36 | 0.804 | 53% | 100% | 5.06 | 7.41 | 0 |
| bedrock:deepseek.v3.2 | model | 22 | 0.735 | 41% | 100% | 5.05 | 68.33 | 0 |
| bedrock:google.gemma-3-27b-it | model | 36 | 0.709 | 36% | 100% | 4.97 | 33.97 | 0 |
| bedrock:openai.gpt-oss-120b-1:0 | model | 36 | 0.706 | 36% | 100% | 3.11 | 11.17 | 0 |
| 6-probe | baseline | 36 | 0.650 | 28% | 100% | 6.00 | 0.00 | 0 |
| bedrock:mistral.mistral-large-3-675b-instruct | model | 36 | 0.590 | 22% | 94% | 0.42 | 5.40 | 0 |
| max-square-ws | baseline | 36 | 0.521 | 8% | 100% | 0.00 | 0.00 | 0 |
| random | baseline | 36 | 0.380 | 8% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. n = tasks with a graded answer; rate-limit/provider failures are excluded and shown as incomplete. Baselines are not model inference.
