# Phase 4c — Retrieval Mode: dense vs BM25 vs hybrid (D5)

Swept: **retrieval mode**.  Pinned: embedder `embeddinggemma` (PROVISIONAL — pending D3), chunker `recursive`, index `flat` (D4).
Dataset: `synthetic_retrieval.jsonl`.
Scored on **250 of 250** items (0 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config          |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:----------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| hybrid_weighted |      0.768 |      0.944 |       0.964 |       0.988 |        0.3504 |              0.5489 | 0.8473 |    0.8369 |   768 |        97.5935 |
| hybrid_rrf      |      0.74  |      0.948 |       0.964 |       0.988 |        0.3416 |              0.5351 | 0.8329 |    0.8301 |   768 |        83.1912 |
| dense           |      0.684 |      0.892 |       0.944 |       0.968 |        0.3328 |              0.5213 | 0.7799 |    0.7828 |   768 |        46.5012 |
| bm25            |      0.692 |      0.876 |       0.92  |       0.952 |        0.3024 |              0.4737 | 0.7774 |    0.7822 |     0 |        26.9866 |

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

**No decision is recorded from this run.** This set ranks configurations; it
does not measure quality. Decisions live in `04c_hybrid_retrieval.md`.

### What it confirms

The ordering matches the golden set exactly - hybrid > bm25 > dense - at 5x
the sample size. That is the validation the synthetic set was built for: it
agrees with human labels about WHICH config is better.

### What it cannot do

Absolute scores run ~13 points higher than golden (hybrid_rrf: 0.9520 here vs
0.8229 there). That gap is the vocabulary leakage documented in
`evaluation/generate_synthetic.py`: questions written while looking at a
passage reuse its wording, and BM25 then matches it trivially. **Never quote
these numbers as retrieval quality.**

The leakage also biases the lexical arm specifically, which is why
`hybrid_weighted`'s apparent win here was rejected - see the golden report.

### Also missing

Every item is single-hop, generated from ONE chunk. Nothing here tests
multi-hop, comparison, synthesis or cross-document retrieval - the four types
the golden set showed we are worst at (0.375-0.467 recall@10). A config that
improved this set while degrading multi-hop would look like an improvement.
