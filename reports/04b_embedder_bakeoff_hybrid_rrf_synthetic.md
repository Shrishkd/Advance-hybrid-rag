# Phase 4b — Embedding Bake-off (D3)

Swept: **embedding model**.  Pinned: chunker `recursive`, index `flat` (D4), retrieval `hybrid_rrf` (D5), backend `ollama`.
Dataset: `synthetic_retrieval.jsonl`.
Scored on **250 of 250** items (0 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config         |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:---------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| bge-m3         |      0.716 |      0.94  |       0.964 |       0.984 |        0.3264 |              0.5113 | 0.8126 |    0.8107 |  1024 |        182.779 |
| embeddinggemma |      0.74  |      0.948 |       0.964 |       0.988 |        0.3416 |              0.5351 | 0.8329 |    0.8301 |   768 |        132.804 |
| mxbai          |      0.684 |      0.928 |       0.956 |       0.972 |        0.3216 |              0.5038 | 0.7894 |    0.7949 |  1024 |        172.442 |
| nomic          |      0.688 |      0.916 |       0.952 |       0.984 |        0.3136 |              0.4912 | 0.7917 |    0.7958 |   768 |        104.564 |

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

**D3 - chosen embedder: `embeddinggemma`.**

### The headline finding is not which model won

It is that **hybrid fusion almost erases the embedder's effect on recall.**

    recall@10, synthetic n=250        dense      hybrid_rrf
      embeddinggemma                  0.944        0.964
      bge-m3                          0.924        0.964
      nomic                           0.888        0.952
      mxbai                           0.884        0.956
      spread (best - worst)           0.060        0.012

Dense-only, embeddinggemma beat mxbai by 6.0 points with **15 wins and 0
losses** across 250 questions - about as clean a result as this project has
produced. Add BM25 and all six pairs become statistically indistinguishable,
with 244-247 ties out of 250. `bge-m3` vs `embeddinggemma` is literally
+0.0000, 2W/2L.

The mechanism is the one D5 already showed: BM25 and dense retrieval fail on
different questions. A weaker dense arm misses passages that BM25 then
supplies, so fusion pulls every configuration up to roughly the same ceiling.
**Hybrid retrieval buys robustness to the embedder choice.**

### But the effect does NOT vanish for ranking

    MRR, hybrid, synthetic n=250
      SIG  embeddinggemma vs mxbai   +0.0434   47W/17L/186T   p=0.0002
      SIG  embeddinggemma vs nomic   +0.0411   45W/21L/184T   p=0.0043
      ns   embeddinggemma vs bge-m3  +0.0202   36W/23L/191T   p=0.1175

Recall asks *did we find it*; MRR asks *how high did it land*. Fusion answers
the first question for us and leaves the second open - the dense arm still
determines the ordering within the fused pool. So the embedder matters, just
for a narrower reason than the dense-only table suggested.

That distinction has a downstream consequence. Since the generator reads the
top few chunks, MRR is closer to what the user experiences than recall@10 is,
which is why a 4-point MRR gap is worth acting on even though the recall gap
is gone.

### Why embeddinggemma over bge-m3

They cannot be separated on quality - p=0.30 dense, p=0.12 hybrid, on n=250.
The tie breaks on cost, and there it is not close:

| | embeddinggemma | bge-m3 |
|---|---:|---:|
| query latency p50, dense | **59 ms** | 114 ms |
| query latency p50, hybrid | **133 ms** | 183 ms |
| dimensions | **768** | 1024 |
| index memory @5.8k chunks | **17.8 MB** | 23.7 MB |
| model size on disk | **621 MB** | 1.2 GB |
| corpus embed time (this CPU) | ~50 min | ~127 min |

`embeddinggemma` is never worse on quality and cheaper on every axis. It also
has the higher point estimate on both metrics in both modes, so the tie-break
is not being made against the direction of the evidence.

### Constraint to record before someone trips over it

`embeddinggemma` has a **2048-token context window**; `bge-m3` has 8192. That
is irrelevant at our 512-token chunk size and irrelevant to parent-child
retrieval, which embeds CHILDREN (256 tokens). It becomes a hard limit the
moment anything embeds a parent chunk directly, because `parent_size` is
**exactly 2048** - sitting on the boundary with no room for a prefix. If D2
changes `parent_size` upward, this decision needs revisiting.

`mxbai`'s 512-token window is a live constraint right now, not a future one:
our chunks are 512 tokens and the model truncates them. Its last-place finish
should be read with that in mind - it is being judged at a chunk size chosen
before it entered the comparison.

### Evidence trail

| run | n | file |
|---|---:|---|
| dense, golden | 48 | `04b_embedder_bakeoff_dense.md` |
| dense, synthetic | 250 | `04b_embedder_bakeoff_dense_synthetic.md` |
| hybrid, golden | 48 | `04b_embedder_bakeoff_hybrid_rrf.md` |
| hybrid, synthetic | 250 | this file |

The golden set agreed on direction in every case but could not reach
significance - consistent with its role: it measures quality honestly at n=48,
the synthetic set supplies the power to rank.

### Still open

- **Quantisation cross-check.** All of the above uses Ollama-served quantised
  weights, which is what production will use. The `st` (Colab fp16) document
  vectors are imported; the matching query vectors need one short notebook
  run. If the ranking holds there, this decision is robust to precision.
- **D5 should be re-run on `embeddinggemma`**, since it was decided pinned to
  `nomic`. Given the recall convergence above, the hybrid advantage over dense
  will likely SHRINK - the better the dense arm, the less BM25 adds.

### Recorded

- `configs/experiment.yaml` -> `embedding.model: embeddinggemma`.
- `plan.md` decision log -> **D3**.
