# Phase 4d — Cross-encoder Reranking (D6)

Swept: **reranker**.  Pinned: stage 1 `hybrid_rrf` (D5) top_k=20, embedder `nomic`, chunker `recursive`, index `flat` (D4).
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config     |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   rerank_ms_p50 |   dim |   query_ms_p50 |
|:-----------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|----------------:|------:|---------------:|
| minilm-l12 |     0.5069 |     0.7674 |      0.8333 |      0.8438 |        0.2708 |              0.3801 | 0.6887 |    0.7147 |          5786.4 |   768 |       6056.45  |
| none       |     0.4653 |     0.7639 |      0.8229 |      0.8438 |        0.275  |              0.386  | 0.651  |    0.693  |           nan   |   768 |        179.905 |
| minilm-l6  |     0.4965 |     0.7431 |      0.7951 |      0.8438 |        0.2792 |              0.3918 | 0.6863 |    0.7063 |          3019.3 |   768 |       3250.86  |

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

**D6 - PROVISIONAL: `minilm-l12` on quality, but the cost may veto it.**
Latency below was measured under CPU contention and must be re-measured on an
idle machine before this is recorded as final.

### The prior is FALSIFIED

`configs/experiment.yaml` predicted reranking would be *"the biggest single
quality win per unit effort."* It is not. Side by side:

| Change | Best delta | Metric |
|---|---:|---|
| **D5** dense -> hybrid_rrf | **+11.8** | recall@10 |
| **D6** none -> minilm-l12 | +4.2 | recall@1 |

Hybrid fusion delivered nearly three times the improvement, for +40 ms instead
of +3,000 ms. The reranker helps, but it is the second-biggest win in Phase 4,
not the first - and it is the most expensive thing we have measured.

### What reranking actually bought

| | recall@1 | recall@5 | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.4653 | **0.7639** | 0.8229 | 0.8438 | 0.6510 | 0.6930 |
| minilm-l6 | 0.4965 | 0.7431 | **0.7951** | 0.8438 | 0.6863 | 0.7063 |
| **minilm-l12** | **0.5069** | 0.7674 | **0.8333** | 0.8438 | **0.6887** | **0.7147** |

The gains are concentrated exactly where a reranker should help - **the top of
the list**. recall@1 +4.2, MRR +3.8. That is the shape predicted by the D5
finding that BM25 has good coverage and noisy ordering: the cross-encoder is
fixing the ordering, which is its job.

### recall@20 is identical for all three, and that is the point

0.8438 in every row, because the reranker REORDERS the stage-1 top-20 rather
than retrieving anything new. **A reranker cannot recover a document stage 1
never found.** Stage 1's recall@20 is the hard ceiling on this entire stage,
and printing it unchanged in the same table keeps that honest.

It also means the reranker's real job is converting recall@20 into recall@1.
Of the 0.8438 available, `none` converts 0.4653 and `minilm-l12` converts
0.5069 - so roughly 40% of what stage 1 finds is still not reaching the top
slot.

### The finding worth keeping: a reranker can make things WORSE

`minilm-l6` **improves recall@1 by +3.1 and degrades recall@10 by -2.8.**

That is not a contradiction, and it is not noise in the usual sense. Reordering
a fixed pool is zero-sum: promoting a document to rank 1 demotes something
else, and a 22M-parameter cross-encoder is confident enough to push a genuinely
relevant chunk from rank 8 down past rank 10. It wins at the very top and loses
at depth.

The practical consequence: **the benefit of a reranker depends on how many
documents the generator actually reads.** Feed the top 3 and minilm-l6 helps.
Feed the top 10 and it hurts. That coupling between D6 and the Phase 5 context
budget was not obvious before this table, and it means the two decisions cannot
be made fully independently.

`minilm-l12` (33M params, 12 layers) does not show the regression - it improves
every metric. Depth buys judgement here.

### Is any of this outside the noise band?

Honestly: **barely, and only for recall@1 and MRR.**

At n=48 our stated noise band is ~5 points. minilm-l12's best gain is +4.2,
which sits just inside it. MRR's +3.8 and nDCG's +2.2 are smaller still. Three
metrics moving in the same direction is more persuasive than any one of them,
but this is nothing like D5's +11.8.

**This is exactly the case the synthetic set exists for.** 250 auto-labelled
questions cost no human time and give the power to tell +4 from 0. D6 should
not be finalised on 48 items.

### Cost: the number that probably decides this

| | rerank p50 | total query p50 |
|---|---:|---:|
| none | - | 180 ms |
| minilm-l6 | 3,019 ms | 3,251 ms |
| minilm-l12 | 5,786 ms | 6,056 ms |

**Measured under contention** - embeddinggemma was embedding the corpus on the
same CPU throughout - so treat these as an upper bound, likely 2-4x inflated.
Even at a quarter, minilm-l12 costs ~1.4 s per query against 180 ms without,
to buy about 4 points at the top of the list.

On a 7.4 GB CPU-only machine this is close to disqualifying. A cross-encoder
scores QUERY-DOCUMENT PAIRS, so nothing can be precomputed: 20 candidates means
20 forward passes per question, every question, forever. That is the structural
reason this stage is expensive and no amount of caching fixes it.

### Open before this is final

1. **Re-measure latency idle.** Currently confounded.
2. **Re-run on the synthetic 250.** +4.2 at n=48 is not a decision.
3. **Re-run after D3.** Stage 1 is pinned to `nomic`; a better embedder changes
   the candidate pool the reranker sees.
4. **Decide jointly with the Phase 5 context budget**, given the top-k coupling
   above.

### Recorded

- `plan.md` decision log -> **D6** (provisional, quality favours `minilm-l12`).
- `configs/experiment.yaml` `rerank.model` LEFT AT `none` until latency is
  re-measured and the synthetic run confirms the gain. Adopting a 1-3 second
  per-query cost on a gain inside the noise band would be exactly the kind of
  unearned decision this project exists to avoid.
