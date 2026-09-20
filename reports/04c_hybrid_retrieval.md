# Phase 4c — Retrieval Mode: dense vs BM25 vs hybrid (D5)

Swept: **retrieval mode**.  Pinned: embedder `embeddinggemma` (PROVISIONAL — pending D3), chunker `recursive`, index `flat` (D4).
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config          |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:----------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| hybrid_weighted |     0.4653 |     0.7431 |      0.8681 |      0.9062 |        0.2875 |              0.4035 | 0.6596 |    0.711  |   768 |        90.0175 |
| hybrid_rrf      |     0.5069 |     0.7743 |      0.8542 |      0.8854 |        0.2833 |              0.3977 | 0.665  |    0.6885 |   768 |       128.081  |
| dense           |     0.4861 |     0.7465 |      0.7986 |      0.8194 |        0.2667 |              0.3743 | 0.653  |    0.6767 |   768 |        52.6049 |
| bm25            |     0.4444 |     0.6597 |      0.7326 |      0.8229 |        0.2292 |              0.3216 | 0.5961 |    0.6164 |     0 |        30.9683 |

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

**D5 - chosen retrieval mode: `hybrid_rrf`. Unchanged. The JUSTIFICATION has
changed completely, and the original headline number is now obsolete.**

Re-run with `embeddinggemma` as the dense arm (D3), replacing the provisional
run pinned to `nomic`.

### A prediction, made before the re-run and confirmed

Stated in advance: *"the hybrid-over-dense margin will shrink, because a
stronger dense arm leaves BM25 less to add."* Measured:

| dense arm | dataset | dense r@10 | hybrid r@10 | gain |
|---|---|---:|---:|---:|
| nomic | golden n=48 | 0.7049 | 0.8229 | **+11.8** |
| embeddinggemma | golden n=48 | 0.7986 | 0.8542 | **+5.6** |
| nomic | synthetic n=250 | 0.8880 | 0.9520 | **+6.4** |
| embeddinggemma | synthetic n=250 | 0.9440 | 0.9640 | **+2.0** |

The gain roughly halves on both datasets. **The +11.8 that justified this
decision was substantially a measure of how weak `nomic` was**, not of how
much hybrid retrieval adds.

### The recall argument no longer survives its own test

    recall@10, synthetic n=250, embeddinggemma
      ns   dense vs hybrid_rrf   -0.0200  CI[-0.0440,+0.0040]  2W/7L/241T  p=0.1797

241 ties out of 250. With a good dense arm, **hybrid does not find
significantly more than dense alone.** If recall@10 were the only metric, D5
would now be "no measurable difference - pick the cheaper one", and dense is
cheaper (47 ms vs 83 ms).

### The ranking argument is what actually carries D5 now

    MRR, synthetic n=250
      SIG  dense vs hybrid_rrf       -0.0529  21W/58L/171T  p<0.0001
      SIG  bm25  vs hybrid_rrf       -0.0554  20W/60L/170T  p<0.0001
      SIG  dense vs hybrid_weighted  -0.0674  20W/52L/178T  p=0.0002

Hybrid beats **both** of its own arms on MRR, decisively, surviving Bonferroni
across six pairs. It is not merely inheriting the better arm - fusing them
orders results better than either ordering alone.

So the honest statement of what hybrid buys, at this dense-arm quality:

    recall  (did we find it)      no measurable gain
    MRR     (how high did it land) +5.3 points, p < 0.0001

That is the same phenomenon D3 found from the other side: fusion converges
recall across embedders while leaving MRR separated. Two experiments, one
mechanism.

**Why that still justifies the cost.** The generator reads the top few chunks,
so where a passage lands matters more to the user than whether it sits
somewhere in the top 20. +36 ms for a 5.3-point MRR gain is worth paying; the
same 36 ms for a recall gain we cannot measure would not be.

### RRF vs weighted: still a tie, still broken on principle

    hybrid_rrf vs hybrid_weighted, synthetic n=250
      recall@10   +0.0000   1W/1L/248T   p=1.0000
      MRR         -0.0144  18W/24L/208T  p=0.4408
      recall@1    -0.0280   5W/12L/233T  p=0.1435

`hybrid_weighted` has the higher point estimate on MRR and recall@1 for the
second time - and for the second time it fails to reach significance. 248 ties
out of 250 on recall@10.

**RRF stands.** `hybrid_weighted` ran at an untuned `w_dense=0.5`; adopting it
would mean adopting a free parameter fitted to nothing, on evidence that has
now twice failed its own test. The no-free-parameter argument is unchanged.

### What this re-run cost, and why it was worth doing

Nothing but compute - the vectors were already cached. Had it been skipped,
the README would still claim hybrid buys +11.8 recall@10, which is now known
to be an artifact of the embedder it was measured with. **A decision made
against a provisional dependency has to be re-made when that dependency
resolves**, even when the decision itself does not change.

### Recorded

- `configs/experiment.yaml` -> `retrieval.mode: hybrid_rrf`, tag `[PROVEN]`
  (was `[PROVEN*]` provisional).
- `plan.md` decision log -> **D5**, no longer provisional.
