# Phase 4b — Embedding Bake-off (D3)

Swept: **embedding model**.  Pinned: chunker `recursive`, index `flat` (D4), retrieval `dense` (D5), backend `ollama`.
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config         |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:---------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| embeddinggemma |     0.4861 |     0.7465 |      0.7986 |      0.8194 |        0.2667 |              0.3743 | 0.653  |    0.6767 |   768 |        62.6644 |
| bge-m3         |     0.4444 |     0.7049 |      0.7222 |      0.8021 |        0.2458 |              0.345  | 0.621  |    0.6342 |  1024 |       134.035  |
| mxbai          |     0.4653 |     0.6944 |      0.7222 |      0.7812 |        0.2708 |              0.3801 | 0.6159 |    0.6349 |  1024 |       158.837  |
| nomic          |     0.4236 |     0.7049 |      0.7049 |      0.7778 |        0.2458 |              0.345  | 0.6014 |    0.6222 |   768 |        65.5674 |

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
