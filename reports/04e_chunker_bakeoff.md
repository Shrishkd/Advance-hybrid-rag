# Phase 4e — Chunking Strategy Bake-off (D2)

Swept: **chunking strategy**.  Pinned: embedder `embeddinggemma`, retrieval `hybrid_rrf` (D5), index `flat` (D4). Parent-child scored on CHILD chunks..
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config       |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   n_chunks |   dim |   query_ms_p50 |
|:-------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|-----------:|------:|---------------:|
| recursive    |     0.5069 |     0.7743 |      0.8542 |      0.8854 |        0.2833 |              0.3977 | 0.665  |    0.6885 |       5796 |   768 |        81.1248 |
| semantic     |     0.434  |     0.8056 |      0.8542 |      0.9271 |        0.2667 |              0.4384 | 0.607  |    0.6778 |       4871 |   768 |        75.1339 |
| parent_child |     0.3958 |     0.7049 |      0.7951 |      0.8681 |        0.2792 |              0.3401 | 0.5899 |    0.6383 |       8944 |   768 |       104.693  |

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

- **Chosen:** _fill in_
- Record in `plan.md`; update `configs/experiment.yaml` and
  upgrade the tag to `[PROVEN]`.
