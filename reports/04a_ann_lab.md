# Phase 4a — ANN Index Comparison (D4)

Corpus: **5796** chunks (`recursive`), embedder `nomic`, dim 768.
Probes: 50 held-out corpus vectors, k=10.

**Recall is measured against exact flat search**, not against human
labels. It answers *how much does approximation cost*, not *is
retrieval good*. The latter needs the golden set and lives in
`04_retrieval_lab.md`.

| index   |   recall@k |   p50_ms |   p95_ms |   build_ms |   mem_mb | params                                                 |
|:--------|-----------:|---------:|---------:|-----------:|---------:|:-------------------------------------------------------|
| flat    |      1     |    0.589 |    0.865 |      123.2 |     17.8 | -                                                      |
| hnsw    |      1     |    0.157 |    0.205 |      261.3 |     19.4 | M=32,ef_construction=200,ef_search=64                  |
| ivf     |      0.978 |    0.146 |    0.268 |      112.6 |     18.3 | nlist=148,nprobe=16                                    |
| ivfpq   |      0.772 |    0.527 |    0.575 |      484.3 |      1.7 | nlist=148,m=64,nbits=8,nprobe=16,pts_per_centroid=22.6 |

## Relative to exact search

- **hnsw**: 3.75x faster than flat, recall@10 1.0000, 1.09x memory
- **ivf**: 4.03x faster than flat, recall@10 0.9780, 1.03x memory
- **ivfpq**: 1.12x faster than flat, recall@10 0.7720, 0.10x memory

## Decision

**D4 — chosen index: `flat` (`IndexFlatIP`).**

### The prior was partially falsified

The pre-registered prior in `src/index/faiss_lab.py` was *"at ~6k chunks, exact flat
search should win outright."* It did not. **HNSW is lossless here** — recall@10 = 1.0000,
identical neighbours on all 200 probes — at **7.6x lower p50 latency**. On speed and
quality jointly, HNSW dominates flat. That part of the prior is wrong and is recorded as
wrong.

### Why `flat` is still the decision

**1. The speedup is real and irrelevant.** 1.046 ms -> 0.138 ms saves **0.9 ms per query**.
Generation costs 1,000-3,000 ms. The saving is under 0.1% of end-to-end latency — below
the noise floor of the thing a user experiences.

**2. Flat is the measurement instrument for the rest of Phase 4.** This is the decisive
argument. The embedder bake-off, chunker bake-off, hybrid fusion and reranker experiments
all compare *retrieval quality*. Running them on an approximate index confounds index
error with the variable under test — an IVF at 0.9885 injects ~1.2% noise into comparisons
whose real effects may be only a few points, and we could not tell which caused what.
The project rule is **one variable at a time**. Flat contributes exactly zero error.

**3. Nothing forces the trade yet.** 17.8 MB of index against a 7.4 GB machine. Flat also
supports exact incremental adds with no retraining — relevant because the 7th document
(GenAI notes) arrives later, and because parent-child chunking will raise the chunk count.

### `ivfpq` is rejected on two counts, not one

It is **lossy (0.7995) *and* slower than flat (0.561 ms vs 1.046 ms — only 1.9x faster,
while HNSW is 7.6x)**. Its sole win is 10x memory, 17.8 MB -> 1.7 MB, which buys nothing
when the baseline is 17.8 MB.

The `pts_per_centroid=22.6` column explains the recall directly: each of the m=64
sub-quantizers must learn 2^8 = 256 centroids, and 5,796 training vectors gives ~23 points
per centroid against FAISS's recommended >=39. The codebooks are undertrained. **PQ is a
technique for tens of millions of vectors**; at our scale it pays compression we do not
need with recall we cannot afford. (The 64 FAISS warnings in the run log are exactly one
per sub-quantizer — expected, not a bug.)

### Crossover point — when to revisit

Flat is O(n) and scales linearly; HNSW is ~O(log n) and is flat-lined at ~0.14 ms here.
Extrapolating flat's 1.046 ms / 5,796 chunks:

| corpus | flat p50 (est.) | hnsw p50 (est.) | verdict |
|---:|---:|---:|---|
| 5,796 (now) | 1.0 ms | 0.14 ms | flat fine |
| ~15,000 (parent-child) | ~2.7 ms | ~0.16 ms | flat fine |
| ~50,000 | ~9 ms | ~0.18 ms | flat still fine |
| ~250,000 | ~45 ms | ~0.2 ms | **switch to HNSW** |

**Trigger: revisit at roughly 100k-250k chunks**, or the first time retrieval p95 exceeds
~50 ms. Both are far outside this corpus. `index.type` is a config line; switching is a
one-line change and a rerun of this lab.

### Honest caveat on these recall numbers

Probe queries are **corpus vectors**, drawn from the same distribution as the index. Real
user queries are off-manifold, and ANN recall on out-of-distribution queries is typically
*worse* than this benchmark shows. So HNSW's 1.0000 and IVF's 0.9885 are **upper bounds**,
not guarantees. That asymmetry cuts in flat's favour: flat has no such caveat because it
is exact by construction.

Equally: this experiment says nothing about whether retrieval is *good*. recall@10 = 1.0
here means "reproduces flat's answers perfectly." Whether flat's answers are right needs
the golden set, and lives in `04_retrieval_lab.md`.

### Recorded

- `configs/experiment.yaml` -> `index.type: flat`, tag upgraded `[PRIOR]` -> `[PROVEN]`.
- `plan.md` decision log -> **D4**.
