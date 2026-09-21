# Phase 6 — `decompose`

Nodes: `['decompose']` · generator `gpt-oss:20b-cloud` · top_n=10 · n=50

| metric | value |
|---|---:|
| usable | 0.74 |
| recall@10 | 0.7257 |
| usable_multi | 0.7143 |
| recall@10_multi | 0.4881 |
| wrong_refusals | 4 |
| mean_docs | 20.0 |
| wall_s_p50 | 11.3 |
| errored | 0 |

## By question type (indicative at n=4-10 per cell)

| type | n | recall@10 | usable |
|---|---:|---:|---:|
| ambiguous | 2 | 0.500 | 1.00 |
| comparison * | 5 | 0.600 | 0.80 |
| definition | 5 | 0.800 | 0.80 |
| explanation | 10 | 0.900 | 0.70 |
| factual | 6 | 0.667 | 0.83 |
| multi_hop * | 4 | 0.250 | 0.50 |
| numerical | 6 | 0.833 | 0.67 |
| specific_source | 5 | 1.000 | 0.60 |
| synthesis * | 5 | 0.567 | 0.80 |
| unanswerable | 2 | - | 1.00 |

\* multi-passage types - the target of Phase 6.

## Decision - NEGATIVE, and attributed to the model, not the technique

Decomposition by **llama3.2:1b** HURT, most on the questions it was built for:
recall@10 on the multi-passage group fell 0.714 -> 0.488 (-22.6), overall 0.854 -> 0.726,
wrong refusals 1 -> 4. It also damaged single-passage questions (factual 1.000 -> 0.667).

**Cause, read directly from its output:**

- It split **49 of 50** questions, ignoring the instruction to leave single-topic
  questions whole. Fragments are worse search queries than the full question.
- **Topic drift and invented entities**: "Markov blanket" became "Markov chain"; Huyen's
  fintech-lending study became "University of California, Berkeley rejected applicants".
  Retrieval then faithfully fetched the wrong subject.
- Degenerate output: the same query repeated three times.

**This run tested "decomposition by a 1B model", not decomposition.** The two were
confounded. The next run changes ONLY the decomposition model (llama3.2:3b, still local,
per the local-model rule for graph nodes) to separate them.
