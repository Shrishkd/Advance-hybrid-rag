# Phase 4e — Chunking Strategy Bake-off (D2)

Swept: **chunking strategy**.  Pinned: embedder `embeddinggemma`, retrieval `hybrid_rrf` (D5), index `flat` (D4). Parent-child scored on CHILD chunks..
Dataset: `synthetic_retrieval.jsonl`.
Scored on **250 of 250** items (0 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config       |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   n_chunks |   dim |   query_ms_p50 |
|:-------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|-----------:|------:|---------------:|
| parent_child |      0.736 |      0.932 |       0.968 |       0.984 |        0.3208 |              0.4173 | 0.8208 |    0.8094 |       8944 |   768 |       110.916  |
| recursive    |      0.74  |      0.948 |       0.964 |       0.988 |        0.3416 |              0.5351 | 0.8329 |    0.8301 |       5796 |   768 |        83.1053 |
| semantic     |      0.7   |      0.916 |       0.936 |       0.972 |        0.2368 |              0.522  | 0.7946 |    0.8095 |       4871 |   768 |        86.8206 |

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

**D2 - chosen chunker: `recursive`.** All three strategies were measured; none
is significantly better than another at retrieval, so the decision is made on
cost and predictability.

### The headline: chunking strategy barely matters here

    synthetic n=250      chunks   recall@10     MRR
      parent_child        8,944     0.9680    0.8208
      recursive           5,796     0.9640    0.8329
      semantic            4,871     0.9360    0.7946

    golden n=48                   recall@10     MRR
      recursive                     0.8542    0.6650
      semantic                      0.8542    0.6070
      parent_child                  0.7951    0.5899

    paired tests, synthetic n=250, 3 pairs, corrected alpha = 0.0167
      ns  recursive    vs parent_child  -0.0040   3W/4L/243T   p=1.0000
      ns  recursive    vs semantic      +0.0280   9W/2L/239T   p=0.0654
      ns  parent_child vs semantic      +0.0320   9W/1L/240T   p=0.0215

**Not one pair reaches significance.** 239-243 ties out of 250 in every
comparison. After three experiments and two datasets, the honest summary is
that on this corpus, with a good embedder and hybrid retrieval, **how you cut
the text is not where the wins are**. D5 moved MRR by 5.3 points and D3 by 4.1;
D2 moves nothing measurable.

That is itself the result. The instinct to tune chunking first is common and,
here, misplaced.

### The pre-registered prior: directionally right, not significant

`split_semantic`'s docstring predicted: *"this LOSES to structure-aware
recursive chunking on our corpus, because our text is already structured - we
have a real TOC. Semantic chunking is a way of guessing boundaries that a
publisher already marked for us."*

Semantic does lose on **every point estimate across both datasets and both
metrics** - recall@10 -2.8, MRR -3.8 on synthetic; MRR -5.8 on golden. The
direction is consistent and matches the reasoning.

But `recursive vs semantic` gives p=0.0654 against a corrected threshold of
0.0167. **The prior is directionally confirmed and statistically unproven.**
Claiming it as a win would be exactly the p-hacking this project has already
caught itself doing once.

### Why `recursive` and not `semantic`, given they tie

`semantic` produces the FEWEST chunks (4,871), which is the cheapest index.
Three things outweigh that:

1. **Build cost.** `recursive` is string arithmetic - seconds. `semantic`
   needed 63,129 sentence-group embeddings before one chunk existed: ~10.9
   hours on this CPU, and it only happened at all because the work was
   offloaded to a T4. That cost recurs every time the corpus changes.
2. **Unbounded chunk size.** median 287 tokens, **max 8,097**. 27.6% of chunks
   exceed 512 tokens and hold 63.6% of all text; 40 chunks exceed
   `embeddinggemma`'s 2048-token window and are silently TRUNCATED. A chunker
   that cannot respect a size budget cannot respect a context budget either,
   and Phase 5 has to fit chunks into a generator prompt.
3. **Lower point estimates on every metric.** Not significant, but not in its
   favour either.

### Why not `parent_child`

Statistically identical to `recursive` (3W/4L/243T, p=1.0000) while producing
**54% more chunks**. On a 7.4 GB machine that is permanent overhead for no
measurable retrieval gain.

**This does not close the parent-child question.** Its premise is *retrieve
small, feed large* - match a sharp 246-token child, hand the generator its
2048-token parent. Retrieval metrics score what was MATCHED, so the second
half is structurally invisible here. **Phase 5 must revisit it** as a context
sufficiency question. The chunks and vectors are built and cached, so that
re-test costs nothing but a run.

### Measurement caveat, quantified

Smaller chunks cover fewer pages, and a hit is scored by page overlap, so
narrow chunks must land more precisely:

    recursive      median 496 tok   1.44 pages covered   61% single-page
    parent_child   median 246 tok   1.24 pages covered   77% single-page
    semantic       median 287 tok   (long-tailed, max 8,097 tok)

A 14% wider net for `recursive`. Real, recorded, and far too small to
manufacture a 243-out-of-250 tie.

### Recorded

- `configs/experiment.yaml` -> `chunking.strategy: recursive`, tag `[PROVEN]`.
- `plan.md` decision log -> **D2**. Phase 4 complete.
