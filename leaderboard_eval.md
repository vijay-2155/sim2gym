# sim2gym leaderboard (AccelGym on SCALE-Sim)

Task set `6f73cea2f0592ee1`

| model | type | n | mean score | solve rate (>=0.95) | feasible | mean sim calls | mean latency (s) | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | baseline | 12 | 1.000 | 100% | 100% | 0.00 | 0.00 | 0 |
| 6-probe | baseline | 12 | 0.702 | 25% | 100% | 6.00 | 0.00 | 0 |
| max-square-ws | baseline | 12 | 0.524 | 0% | 100% | 0.00 | 0.00 | 0 |
| random | baseline | 12 | 0.458 | 8% | 100% | 0.00 | 0.00 | 0 |

Score = optimum cycles / answer cycles (exhaustive optimum, real SCALE-Sim), 0 if infeasible. Baselines are not model inference.
