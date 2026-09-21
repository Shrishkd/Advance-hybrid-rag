# Phase 6 — `baseline`

Nodes: `none (baseline)` · generator `gpt-oss:20b-cloud` · top_n=10 · n=50

| metric | value |
|---|---:|
| usable | 0.8 |
| recall@10 | 0.8542 |
| usable_multi | 0.7143 |
| recall@10_multi | 0.7143 |
| wrong_refusals | 1 |
| mean_docs | 20.0 |
| wall_s_p50 | 0.2 |

## By question type (indicative at n=4-10 per cell)

| type | n | recall@10 | usable |
|---|---:|---:|---:|
| ambiguous | 2 | 0.500 | 1.00 |
| comparison * | 5 | 0.900 | 0.60 |
| definition | 5 | 1.000 | 0.80 |
| explanation | 10 | 0.900 | 0.90 |
| factual | 6 | 1.000 | 0.83 |
| multi_hop * | 4 | 0.500 | 0.75 |
| numerical | 6 | 0.833 | 0.67 |
| specific_source | 5 | 1.000 | 0.80 |
| synthesis * | 5 | 0.700 | 0.80 |
| unanswerable | 2 | - | 1.00 |

\* multi-passage types - the target of Phase 6.
