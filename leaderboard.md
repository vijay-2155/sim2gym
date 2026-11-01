# sim2gym leaderboard (AccelGym on SCALE-Sim)

Task set `903fb5045d2d4d26`

| model | type | n | mean score | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | baseline | 36 | 1.000 | 100% | 100% | 0.00 | 0.00 | 0 |
| 6-probe | baseline | 36 | 0.650 | 28% | 100% | 6.00 | 0.00 | 0 |
| max-square-ws | baseline | 36 | 0.521 | 8% | 100% | 0.00 | 0.00 | 0 |
| random | baseline | 36 | 0.380 | 8% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. Baselines are not model inference.
