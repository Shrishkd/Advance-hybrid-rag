# Phase 4b — Embedding Bake-off (D3)

Swept: **embedding model**.  Pinned: chunker `recursive`, index `flat` (D4), retrieval `dense` (D5), backend `ollama`.
Dataset: `synthetic_retrieval.jsonl`.
Scored on **250 of 250** items (0 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config         |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:---------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| embeddinggemma |      0.684 |      0.892 |       0.944 |       0.968 |        0.3328 |              0.5213 | 0.7799 |    0.7828 |   768 |        46.71   |
| bge-m3         |      0.648 |      0.872 |       0.924 |       0.968 |        0.3136 |              0.4912 | 0.7531 |    0.7593 |  1024 |        92.3956 |
| nomic          |      0.632 |      0.84  |       0.888 |       0.94  |        0.2928 |              0.4586 | 0.7226 |    0.7314 |   768 |        40.2511 |
| mxbai          |      0.64  |      0.832 |       0.884 |       0.928 |        0.2904 |              0.4549 | 0.7274 |    0.7356 |  1024 |       110.096  |

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
