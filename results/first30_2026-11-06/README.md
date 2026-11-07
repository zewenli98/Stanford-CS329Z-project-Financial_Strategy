# First-30 comparison, 2026-11-06

Sessions 2026-07-16 through 2026-08-26. The label is regular-session `(high - low) / low`.

The market ranker uses the frozen 20-feature weights in `configs/screening_research100_llm_live.yaml`. Residual scoring was off, so ranking language-model research did not change the published order.

| Arm | NDCG@10 | Notes |
|---|---:|---|
| Frozen market ranker | 0.7817 | `market_model_report.json` |
| Ranking language model | 0.7817 | Same top 10 on all 30 days. Paired delta 0. |
| Adaptive historical-range baseline | 0.7800 | No model call |
| Adaptive scenario | 0.7799 | Top 10 changed on 3 days. Paired NDCG delta −0.000099 |

Recorded spend was $2.74 for the ranking run and $0.73 for adaptive scenario. Raw run directories, recordings, and logs stay local under `runs/`.
