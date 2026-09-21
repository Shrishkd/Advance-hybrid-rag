# Phase 6 — `decompose3b_keep`

Nodes: `['decompose', 'keep_original']` · generator `gpt-oss:20b-cloud` · transform `llama3.2:3b` · top_n=10 · n=50

| metric | value |
|---|---:|
| usable | 0.72 |
| recall@10 | 0.7743 |
| usable_multi | 0.5 |
| recall@10_multi | 0.6548 |
| wrong_refusals | 4 |
| mean_docs | 20.0 |
| wall_s_p50 | 0.3 |
| errored | 0 |

## By question type (indicative at n=4-10 per cell)

| type | n | recall@10 | usable |
|---|---:|---:|---:|
| ambiguous | 2 | 0.500 | 1.00 |
| comparison * | 5 | 0.800 | 0.60 |
| definition | 5 | 0.800 | 1.00 |
| explanation | 10 | 0.900 | 0.60 |
| factual | 6 | 0.833 | 1.00 |
| multi_hop * | 4 | 0.500 | 0.50 |
| numerical | 6 | 0.833 | 0.83 |
| specific_source | 5 | 0.800 | 0.60 |
| synthesis * | 5 | 0.633 | 0.40 |
| unanswerable | 2 | - | 1.00 |

\* multi-passage types - the target of Phase 6.
