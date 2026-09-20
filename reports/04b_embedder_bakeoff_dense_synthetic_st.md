# Phase 4b — Embedding Bake-off (D3)

Swept: **embedding model**.  Pinned: chunker `recursive`, index `flat` (D4), retrieval `dense` (D5), backend `st`.
Dataset: `synthetic_retrieval.jsonl`.
Scored on **250 of 250** items (0 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config         |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:---------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| embeddinggemma |      0.684 |      0.892 |       0.944 |       0.968 |        0.3328 |              0.5213 | 0.7798 |    0.7828 |   768 |         1.0981 |
| bge-m3         |      0.648 |      0.872 |       0.924 |       0.968 |        0.3144 |              0.4925 | 0.7533 |    0.7593 |  1024 |         1.346  |
| nomic          |      0.632 |      0.84  |       0.888 |       0.94  |        0.2928 |              0.4586 | 0.7222 |    0.7311 |   768 |         1.2988 |
| mxbai          |      0.64  |      0.832 |       0.884 |       0.928 |        0.2904 |              0.4549 | 0.7271 |    0.7355 |  1024 |         1.4248 |

## Caveats that belong next to these numbers

- **n is small.** With ~48 scored items a 5-point gap is inside the
  noise band. Treat anything under ~5 points as a tie and break it on
  cost, context window, or latency instead.
- **17 of 50 questions name their source book** (e.g. "According to
  Huyen…"). Author names do not appear in chunk text, so that hint
  pays off only once metadata filtering exists — it is dead weight in
  the query here, and slightly understates every embedder.
- **`mxbai` has a 512-token context window** against 512-token chunks
  plus a query prefix, so it truncates. That is a real property of the
  model at our chunk size, not a bug — but it means mxbai is being
  judged on a chunk size chosen before it was in the running.

## Decision

**No new decision. This run exists to falsify D3, and it fails to.**

### The question

Every D3 number was produced with Ollama-served **quantised GGUF** weights.
The obvious objection is that the ranking might be an artifact of
quantisation - that a model tolerant of low precision looks better than a more
capable model damaged by it. If so, the production config would be chosen for
the wrong reason.

This run repeats the dense sweep against the SAME chunks embedded at **fp16 on
a Colab T4**, with queries embedded by the same fp16 weights. One variable
changes: numeric precision.

### Result: the ranking is unchanged, and so are the numbers

| model | recall@10 ollama | recall@10 st | MRR ollama | MRR st |
|---|---:|---:|---:|---:|
| embeddinggemma | 0.9440 | 0.9440 | 0.7799 | 0.7798 |
| bge-m3 | 0.9240 | 0.9240 | 0.7531 | 0.7533 |
| nomic | 0.8880 | 0.8880 | 0.7226 | 0.7222 |
| mxbai | 0.8840 | 0.8840 | 0.7274 | 0.7271 |

**recall@10 is identical to four decimal places for all four models.** MRR
moves only in the fourth decimal, in both directions.

### Why identical results are not evidence of a bug

Identical numbers from two pipelines should always be suspected first. Two
checks say this is real:

1. The arrays are **not equal** (`np.array_equal` is False for all four), and
   query latency differs by two orders of magnitude - 1-2 ms for cached fp16
   query vectors versus 40-110 ms for live Ollama inference. Different code
   paths, different data.
2. **MRR does move.** If the st run were secretly reading ollama vectors, MRR
   would be identical too. The rankings differ slightly; the top-10 membership
   does not.

Direct measurement of the vectors explains it. Cosine between the quantised
and fp16 embedding of the SAME chunk:

    nomic           mean 1.00000   min 0.99725
    bge-m3          mean 0.99999   min 0.99964
    embeddinggemma  mean 0.99999   min 0.99654
    mxbai           mean 1.00000   min 0.99758

A perturbation that small reshuffles neighbours at the margin - which is what
MRR picks up - but essentially never pushes a document across the rank-10
boundary, which is what recall@10 measures.

### A claim of mine, falsified

When `src/embed/st_embedder.py` was written its docstring asserted that mixing
backends produces "a systematic quantisation error that varies by model". That
was a guess stated as fact, and the table above shows it overstated the effect
by a wide margin. Mixing fp16 documents with quantised queries would have been
very nearly harmless on this corpus.

The separation is still correct - its cost was zero, and the size of the error
was unknown until measured. But the reasoning offered at the time was wrong,
and the docstring now carries these numbers instead of the guess.

### What this buys

- **D3 is robust to precision.** `embeddinggemma` wins on both backends by the
  same margins.
- **Quantisation is free for retrieval at this scale.** The production path
  serves quantised weights through Ollama and gives up nothing measurable.
  That removes fp16 serving from the list of things worth engineering.
- **Colab remains useful for THROUGHPUT, not fidelity.** ~50 minutes of CPU
  per model versus a few minutes on a T4, for vectors that behave identically.
