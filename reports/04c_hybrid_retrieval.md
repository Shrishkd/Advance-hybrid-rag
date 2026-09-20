# Phase 4c — Retrieval Mode: dense vs BM25 vs hybrid (D5)

Swept: **retrieval mode**.  Pinned: embedder `nomic` (PROVISIONAL — pending D3), chunker `recursive`, index `flat` (D4).
Dataset: `golden_50.jsonl`.
Scored on **48 of 50** items (2 unanswerable excluded — they have no ground-truth contexts, so recall on them is 0.0 by construction and would apply a constant penalty to every row).

Metrics are judge-free: computed from labelled `(book, page_range)`
overlap. Index pinned to `flat` per **D4**, so no approximation error
is mixed into these differences.

| config          |   recall@1 |   recall@5 |   recall@10 |   recall@20 |   precision@5 |   prec@5_vs_ceiling |    mrr |   ndcg@10 |   dim |   query_ms_p50 |
|:----------------|-----------:|-----------:|------------:|------------:|--------------:|--------------------:|-------:|----------:|------:|---------------:|
| hybrid_rrf      |     0.4653 |     0.7639 |      0.8229 |      0.8438 |        0.275  |              0.386  | 0.651  |    0.693  |   768 |        275.141 |
| hybrid_weighted |     0.4757 |     0.7431 |      0.816  |      0.8542 |        0.2708 |              0.3801 | 0.6521 |    0.6773 |   768 |        284.477 |
| bm25            |     0.4444 |     0.6597 |      0.7326 |      0.8229 |        0.2292 |              0.3216 | 0.5961 |    0.6164 |     0 |        109.019 |
| dense           |     0.4236 |     0.7049 |      0.7049 |      0.7778 |        0.2458 |              0.345  | 0.6014 |    0.6222 |   768 |        210.845 |

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

**D5 - chosen retrieval mode: `hybrid_rrf`.** PROVISIONAL: pinned to `nomic`
pending D3. Re-run after the embedder is chosen.

### The prior held, and by a wide margin

Pre-registered in `configs/experiment.yaml`: *"hybrid wins clearly - textbooks
are full of exact terms (ReLU, Adam, BLEU) that embeddings blur but BM25
nails."* Measured:

| | recall@10 | vs dense |
|---|---:|---:|
| dense | 0.7049 | - |
| bm25 | 0.7326 | +2.8 |
| **hybrid_rrf** | **0.8229** | **+11.8** |
| hybrid_weighted | 0.8160 | +11.1 |

**+11.8 points at n=48.** Our stated noise band is ~5 points, so this is the
first Phase 4 effect comfortably clear of it. MRR (+5.0) and nDCG@10 (+7.1)
move in the same direction, which matters: a gain that appeared in recall but
not in ranking quality would suggest we were merely retrieving more, not
better.

### The unexpected finding: BM25 alone beats dense

`bm25` scores 0.7326 recall@10 against dense's 0.7049 - a lexical method from
1994, no embedding model, no GPU, a 1.3-second index build, beating a
768-dimensional neural embedder on our corpus. It is also **2x faster**
(51 ms vs 104 ms p50).

Technical textbooks are dense with tokens that must match literally - `ReLU`,
`AdaGrad`, `L-BFGS`, `CKY`, `BLEU`. Dense embeddings place `Adam` near
`RMSProp` near `SGD`, which is correct semantics and wrong retrieval when the
question names one of them.

It does not mean dense is useless: fusing the two beats either alone by ~9-12
points, so each arm finds documents the other misses.

### The mechanism, visible in the numbers

Dense has **recall@5 == recall@10 == 0.7049**. Ranks 6-10 contribute nothing:
dense either surfaces the passage in its top 5 or never. Hybrid breaks that
flat spot - 0.7639 at k=5 rising to 0.8229 at k=10 - because BM25 contributes
documents dense had not ranked anywhere near the top. That is what RRF is
designed to reward, and it is the clearest evidence here that the two signals
are complementary rather than redundant.

The same asymmetry appears from the other side: BM25 **loses** at k=5 (0.6597
vs dense 0.7049) and **wins** at k=10. Lexical match has better coverage but
noisier ordering - an exact term hit can land on a passing mention. Dense
orders better but has a hard coverage ceiling. That is precisely the shape of
problem a cross-encoder reranker (D6) is meant to fix, and it is why the
reranker belongs after the FUSED pool rather than after either arm.

### RRF vs weighted fusion: a genuine tie, broken on principle

| | recall@10 | recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| hybrid_rrf | **0.8229** | 0.8438 | 0.6510 | **0.6930** |
| hybrid_weighted | 0.8160 | **0.8542** | **0.6521** | 0.6773 |

They trade wins across four metrics and no gap exceeds 1.1 points. At n=48
this is a tie, and calling either "the winner" would be false precision.

**RRF is chosen because it has no free parameter.** `hybrid_weighted` ran at
`w_dense=0.5`, a value nobody tuned. Tuning it on 48 questions would fit a
weight to this specific sample, and a project whose premise is that decisions
must be earned should not adopt a knob it cannot justify. RRF's k=60 is the
published default, adopted knowingly and unchanged.

### Cost

Hybrid costs **+40 ms p50** over dense (144 ms vs 104 ms) plus one in-memory
index built in ~1.3 s. Against 1,000-3,000 ms of generation, that is noise.
Accepted without reservation.

### What this does NOT settle

- **Pinned to `nomic`.** If D3 picks a different embedder the dense arm gets
  stronger and this margin may shrink. Re-run required.
- **bge-m3 emits sparse and ColBERT vectors from one model**, which could
  replace BM25 with a learned lexical signal. Ollama serves dense only, so
  that needs FlagEmbedding on Colab - deferred, flagged in `plan.md`.
- **No reranker yet (D6).** hybrid's recall@20 = 0.8438 is the hard ceiling on
  anything reranking can deliver.

### Confirmed statistically, after the fact

The "genuine tie" above was originally a judgement call from eyeballing four
metrics. It now has a paired test behind it (`evaluation/significance.py`):

    hybrid_rrf vs hybrid_weighted, recall@10, n=48 golden
      difference   +0.0069   95% CI [-0.0347, +0.0486]
      sign test    3W / 2L / 43T    p = 1.0000

**43 ties out of 48.** The two fusion methods return the same result on 90% of
questions. "Tie" was the right call.

The ~5-point noise band quoted throughout this project is also now known to be
the WRONG tool. It is the standard error of an unpaired proportion, and these
comparisons are paired - every config answers the same questions. The paired
view is much sharper, and it is what the 250-item run below uses.

### Re-tested on the synthetic 250, where weighted appeared to win

On `synthetic_retrieval.jsonl` (n=250) `hybrid_weighted` led on every headline
number, which looked like grounds to revisit this decision. Tested properly:

| metric | diff | 95% CI | sign test | p |
|---|---:|---|---|---:|
| recall@10 | +0.0040 | [-0.0120, +0.0240] | 3W/2L/**245T** | 1.000 |
| MRR | +0.0217 | [+0.0020, +0.0414] | 33W/18L/199T | 0.049 |
| recall@1 | +0.0280 | [-0.0080, +0.0640] | 13W/6L/231T | 0.167 |

One metric of three clears 0.05 - and three metrics were tested. Reporting the
one that passed is p-hacking. Bonferroni puts the threshold at 0.0167, and
MRR's p=0.0306 **does not survive it**.

Two further reasons not to flip on this evidence even if it had survived:

1. **245 of 250 ties on recall@10.** Whatever separates these methods is
   confined to ordering within an almost identical result set.
2. **The synthetic set's known bias points exactly this way.** Its questions
   were written while looking at the passage, which flatters lexical
   retrieval - and weighted fusion is precisely the method that amplifies a
   confident lexical arm, because it blends raw scores rather than ranks. A
   win here is what the bias predicts, not independent evidence.

**D5 unchanged: `hybrid_rrf`.** The no-free-parameter argument remains decisive.

### Recorded

- `configs/experiment.yaml` -> `retrieval.mode: hybrid_rrf`.
- `plan.md` decision log -> **D5** (provisional).
