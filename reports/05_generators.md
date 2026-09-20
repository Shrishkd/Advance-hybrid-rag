# Phase 5 — Generator Comparison (D9)

50 golden items · prompt `answer_v0` · top_n=6 · budget=3000 tokens

Retrieval FROZEN at the Phase 4 winner (`embeddinggemma` + `hybrid_rrf` + `recursive` + `flat`) and computed once, so every row differs only by generator.

**No LLM judge.** Every column is deterministic and free. Whether an answer is CORRECT, or whether a cited claim is actually supported by the chunk it points at, is Phase 8's question.

| config            |   grounded_rate |   has_citations_rate |   fabrication_rate |   n_fabricated_total | correct_refusals   |   wrong_refusals |   mean_sources_cited |   answer_words_median |   latency_p50_ms |   latency_p95_ms |   errors |
|:------------------|----------------:|---------------------:|-------------------:|---------------------:|:-------------------|-----------------:|---------------------:|----------------------:|-----------------:|-----------------:|---------:|
| gpt-oss:20b-cloud |             0.6 |                  0.6 |                  0 |                    0 | 2/2                |                4 |                 1.02 |                    23 |             9256 |            24258 |        1 |

## Reading this table

- **grounded_rate** — cited at least once AND every citation resolves. The headline number.
- **fabrication_rate alone is a trap**: a model that never cites scores a perfect 0.0. Always read it beside `has_citations_rate`.
- **correct_refusals** — of the items whose `expected_behaviour` is `refuse`. The 2 `ambiguous` items expect `clarify`, not refusal, and are counted as wrong refusals if declined.
- **wrong_refusals** — declined a question the corpus can answer. The expensive failure: an unhelpful system that looks safe.

## Decision

- **Chosen generator:** _fill in_
- Record as **D9** in `plan.md`; set `generation.model_role` in
  `configs/experiment.yaml`.
