# Phase 4b — Embedding Bake-off (D3)

Swept: **embedding model**.  Pinned: chunker `recursive`, index `flat` (D4), retrieval `hybrid_rrf` (D5), backend `ollama`.
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config         |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:---------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| embeddinggemma |     0.5069 |     0.7743 |      0.8542 |      0.8854 |        0.2833 |              0.3977 | 0.665  |    0.6885 |   768 |        170.655 |
| nomic          |     0.4653 |     0.7639 |      0.8229 |      0.8438 |        0.275  |              0.386  | 0.651  |    0.693  |   768 |        119.411 |
| bge-m3         |     0.5486 |     0.7604 |      0.8125 |      0.8333 |        0.2875 |              0.4035 | 0.6856 |    0.7042 |  1024 |        204.581 |
| mxbai          |     0.4965 |     0.7396 |      0.8125 |      0.8542 |        0.275  |              0.386  | 0.6512 |    0.6694 |  1024 |        214.324 |

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
