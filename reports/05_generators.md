# Phase 5 — Generator Comparison (D9)

50 golden items · prompt `answer_v0` · top_n=6 · budget=3000 tokens

Retrieval FROZEN at the Phase 4 winner (`embeddinggemma` + `hybrid_rrf` + `recursive` + `flat`) and computed once, so every row differs only by generator.

**No LLM judge.** Every column is deterministic and free. Whether an answer is CORRECT, or whether a cited claim is actually supported by the chunk it points at, is Phase 8's question.

| config            |   usable_rate |   grounded_rate |   has_citations_rate |   fabrication_rate |   n_fabricated_total | correct_refusals   |   wrong_refusals |   mean_sources_cited |   answer_words_median |   over_length_rate |   leaked_reasoning_rate |   latency_p50_ms |   latency_p95_ms |   n_timed |   errors |   faith_supported_rate | faith_answers_with_unsupported   |
|:------------------|--------------:|----------------:|---------------------:|-------------------:|---------------------:|:-------------------|-----------------:|---------------------:|----------------------:|-------------------:|------------------------:|-----------------:|-----------------:|----------:|---------:|-----------------------:|:---------------------------------|
| llama3.2:3b       |          1    |            0.96 |                 0.96 |             0      |                    0 | 2/2                |                0 |                 1.67 |                    51 |               0    |                       0 |              nan |              nan |         0 |        0 |                 0.7014 | 14/48                            |
| gpt-oss:20b-cloud |          0.76 |            0.72 |                 0.72 |             0      |                    0 | 2/2                |                5 |                 1.19 |                    48 |               0    |                       0 |            12393 |            13011 |         2 |        0 |                 0.8727 | 1/36                             |
| phi4-mini         |          0.68 |            0.68 |                 0.72 |             0.0164 |                   10 | 2/2                |                5 |                 2.29 |                    72 |               0.06 |                       0 |              nan |              nan |         0 |        0 |                 0.549  | 20/34                            |

## Reading this table

- **usable_rate** — THE HEADLINE. Grounded AND no leaked reasoning AND within length. The only column a user-facing system can be judged on.
- **grounded_rate** — cited at least once AND every citation resolves. Necessary, not sufficient: it scored qwen3:4b's truncated reasoning monologues at 14/14.
- **fabrication_rate alone is a trap**: a model that never cites scores a perfect 0.0. Always read it beside `has_citations_rate`.
- **correct_refusals** — of the items whose `expected_behaviour` is `refuse`. The 2 `ambiguous` items expect `clarify`, not refusal, and are counted as wrong refusals if declined.
- **wrong_refusals** — declined a question the corpus can answer. The expensive failure: an unhelpful system that looks safe.
- **leaked_reasoning_rate / over_length_rate** — FORMAT integrity, which citation integrity cannot see. A reasoning trace that says 'We are to cite [S1]' cites S1 and scores as grounded. Any model with a non-zero leak rate has an inflated `grounded_rate`.

## Decision

**D9 - chosen generator: `gpt-oss:20b-cloud`.** Decided by Shrish 2026-09-21, choosing
the free same-family judge over paying for an independent one.

### The obligation this decision creates

With gpt-oss generating, the only free judge (gpt-oss:120b) is the same family. Shrish
accepted that on one condition, which is now **binding on Phase 8**:

> Hand-label ~15 answers and measure the judge's agreement with them BEFORE any judged
> metric is reported. That measures self-preference bias directly instead of assuming it
> away. If agreement on gpt-oss answers is materially higher than on llama/phi4 answers,
> the judge is flattering its own family and Phase 8's numbers need that correction.

Until that validation exists, every faithfulness number for gpt-oss:20b carries the
same-family caveat.

### The four views, and how the ranking changed at each

| model | usable | faithfulness | cites dumped up front | latency p50 (clean) |
|---|---:|---:|---:|---:|
| llama3.2:3b | **1.00** | 0.70 | 37/48 (77%) | 61.2 s |
| gpt-oss:20b-cloud | 0.76 | **0.87**\* | **0/36** (0%) | **6.6 s** |
| phi4-mini | 0.68 | 0.55 | 19/35 (54%) | 73.0 s |
| qwen3:4b | 0.00 | - | - | ~330 s |

\* judged by gpt-oss:120b-cloud - SAME FAMILY; see below.

Each layer of measurement changed who looked best, which is the point of this phase:

1. **grounded_rate** said qwen3:4b was perfect (14/14). It was 14 truncated reasoning
   monologues that "cited" [S1] while never reaching an answer.
2. **usable_rate** (grounded + clean format + correct refusal) said llama3.2:3b was
   perfect, 50/50, beating gpt-oss 12W/0L (p=0.0005).
3. **faithfulness** (does the cited source SUPPORT the claim?) showed 29% of llama's
   answers carry at least one unsupported citation.
4. **citation placement**, measured with no judge, explained why.

### llama's failure mode: citation dumping

In 77% of answers llama3.2:3b prepends every source tag - "[S1][S2][S3] The
regularization term..." - instead of attaching each to its claim. That violates prompt
rule 2 and attaches sources that support nothing. Reading four flagged answers by hand,
the judge was clearly right on three and debatable on one.

### Why the gpt-oss faithfulness number is credible despite the same-family judge

A judge must not grade its own model family unchecked. gpt-oss:120b judging gpt-oss:20b is the
self-preference case, and its 0.87 alone would not be trustworthy.

But placement is measured by regex, not by any model: **gpt-oss attaches citations to
claims in 36/36 answers and dumps in 0**. That is an independent mechanism for higher
faithfulness which self-preference bias cannot manufacture. The number is corroborated,
not proven.

### gpt-oss's failure mode is conservative

5 wrong refusals and 7 uncited answers - it DECLINES rather than misattributes. For a
RAG system the inverse failure (a confident answer citing a source that does not say
that) is the more damaging one: it looks exactly like a correct answer.

### The constraint that makes this a decision rather than a result

Only the gpt-oss family is free on Ollama Cloud (`glm-5.3-flash` returns "not included
in your free usage", tested 2026-09-21). So:

- **gpt-oss generates** -> Phase 8 has no free independent judge.
- **llama generates** -> gpt-oss:120b is a clean judge, but answers take a minute each
  and 29% carry an unsupported citation.

### Still unmeasured, and cheap

- **Prompt v1** with an explicit inline-citation example may fix llama's dumping, and a
  clarified refusal rule may cut gpt-oss's wrong refusals. Both are prompt changes.
- **top_n sweep** - latency is prefill-dominated, so fewer chunks cuts llama's 61 s.
