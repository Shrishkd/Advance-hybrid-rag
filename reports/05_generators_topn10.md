# Phase 5 — Generator Comparison (D9)

50 golden items · prompt `answer_v0` · top_n=10 · budget=5300 tokens

Retrieval FROZEN at the Phase 4 winner (`embeddinggemma` + `hybrid_rrf` + `recursive` + `flat`) and computed once, so every row differs only by generator.

**No LLM judge.** Every column is deterministic and free. Whether an answer is CORRECT, or whether a cited claim is actually supported by the chunk it points at, is Phase 8's question.

| config            |   usable_rate |   grounded_rate |   has_citations_rate |   fabrication_rate |   n_fabricated_total | correct_refusals   |   wrong_refusals |   mean_sources_cited |   answer_words_median |   over_length_rate |   leaked_reasoning_rate |   latency_p50_ms |   latency_p95_ms |   n_timed |   errors |   faith_supported_rate | faith_answers_with_unsupported   |
|:------------------|--------------:|----------------:|---------------------:|-------------------:|---------------------:|:-------------------|-----------------:|---------------------:|----------------------:|-------------------:|------------------------:|-----------------:|-----------------:|----------:|---------:|-----------------------:|:---------------------------------|
| gpt-oss:20b-cloud |           0.8 |            0.76 |                 0.76 |                  0 |                    0 | 2/2                |                2 |                 1.54 |                    51 |                  0 |                       0 |             7805 |            25766 |        50 |        0 |                 0.8711 | 2/38                             |

## Reading this table

- **usable_rate** — THE HEADLINE. Grounded AND no leaked reasoning AND within length. The only column a user-facing system can be judged on.
- **grounded_rate** — cited at least once AND every citation resolves. Necessary, not sufficient: it scored qwen3:4b's truncated reasoning monologues at 14/14.
- **fabrication_rate alone is a trap**: a model that never cites scores a perfect 0.0. Always read it beside `has_citations_rate`.
- **correct_refusals** — of the items whose `expected_behaviour` is `refuse`. The 2 `ambiguous` items expect `clarify`, not refusal, and are counted as wrong refusals if declined.
- **wrong_refusals** — declined a question the corpus can answer. The expensive failure: an unhelpful system that looks safe.
- **leaked_reasoning_rate / over_length_rate** — FORMAT integrity, which citation integrity cannot see. A reasoning trace that says 'We are to cite [S1]' cites S1 and scores as grounded. Any model with a non-zero leak rate has an inflated `grounded_rate`.

## Decision

- **Chosen generator:** _fill in_
- Record as **D9** in `plan.md`; set `generation.model_role` in
  `configs/experiment.yaml`.
