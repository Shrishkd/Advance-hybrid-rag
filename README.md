# Advanced Hybrid RAG Chatbot

A retrieval-augmented chatbot over six machine-learning textbooks — built so that **every
architectural decision is measured rather than assumed**.

Most RAG projects pick a chunker, an embedder and a vector store on the first day and never
revisit them. This one treats each of those as an open question, benchmarks the alternatives
against a hand-labelled golden dataset, and ships the winner. The tables below are the
project; the chatbot is what falls out of them.

> **Status:** Phase 4 of 10 — retrieval lab. Three decisions measured (D1, D4, D5), three in
> flight. Empty cells below are honest: they mean *not yet measured*. See
> [`plan.md`](plan.md) for the live status board.

---

## Corpus

| Book | Author |
|---|---|
| Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow | Aurélien Géron |
| Deep Learning | Goodfellow, Bengio, Courville |
| Speech and Language Processing | Jurafsky & Martin |
| Pattern Recognition and Machine Learning | Christopher Bishop |
| Designing Machine Learning Systems | Chip Huyen |
| Build a Large Language Model (From Scratch) | Sebastian Raschka |

~3,500 pages of equation-dense technical text. PDFs are **not** distributed with this repo.

---

## Results

*Populated as each phase completes. Empty cells are honest — they mean "not yet measured".*

### Retrieval mode — D5 · [full report](reports/04c_hybrid_retrieval.md)

Measured on 48 hand-labelled questions. Embedder `nomic`, chunker `recursive`,
index `flat`. **No LLM judge** — every number comes from labelled
`(book, page_range)` overlap, so these are reproducible to the fourth decimal.

| Config | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | p50 |
|---|---:|---:|---:|---:|---:|---:|
| **hybrid_rrf** ✅ | 0.7639 | **0.8229** | 0.8438 | 0.6510 | **0.6930** | 144 ms |
| hybrid_weighted | 0.7431 | 0.8160 | **0.8542** | **0.6521** | 0.6773 | 153 ms |
| bm25 | 0.6597 | 0.7326 | 0.8229 | 0.5961 | 0.6164 | **51 ms** |
| dense | 0.7049 | 0.7049 | 0.7778 | 0.6014 | 0.6222 | 104 ms |

**Hybrid beats dense by +11.8 points of recall@10** — the first effect in this
project comfortably clear of the ~5-point noise band at n=48.

**The surprise: BM25 alone beats dense.** A lexical method from 1994, with no
embedding model and a 1.3-second index build, outscores a 768-dimensional
neural embedder — and runs twice as fast. Technical textbooks are full of
tokens that must match *literally* (`ReLU`, `AdaGrad`, `L-BFGS`, `CKY`). Dense
embeddings place `Adam` near `RMSProp` near `SGD`: correct semantics, wrong
retrieval when the question names one of them.

**Why fusing them works, visible in the table:** dense has `recall@5 ==
recall@10` — ranks 6–10 contribute *nothing*; it either finds the passage in
the top 5 or never. BM25 shows the mirror image, losing at k=5 and winning at
k=10: better coverage, noisier ordering. Each arm fails where the other
succeeds, which is exactly the condition under which fusion pays.

RRF vs weighted fusion is a **genuine tie** (no gap above 1.1 points across
four metrics). RRF wins on principle: it has no free parameter, while the
weighted variant ran at an untuned `w_dense=0.5`.

### Reranking — D6 · [full report](reports/04d_reranking.md)

Cross-encoder over the fused top-20. Stage 1 pinned to `hybrid_rrf`.

| Reranker | Recall@1 | Recall@10 | Recall@20 | MRR | nDCG@10 | Rerank p50 |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.4653 | 0.8229 | 0.8438 | 0.6510 | 0.6930 | — |
| minilm-l6 | 0.4965 | *0.7951* | 0.8438 | 0.6863 | 0.7063 | 3,019 ms |
| minilm-l12 | **0.5069** | **0.8333** | 0.8438 | **0.6887** | **0.7147** | 5,786 ms |

**A second prior falsified.** We predicted reranking would be the biggest
single quality win. It isn't — hybrid fusion (+11.8 recall@10) beat it by
nearly 3× and cost +40 ms instead of +3,000 ms. **Not adopted**: a +4.2 gain
sits inside the ~5-point noise band at n=48, and 1–6 seconds per query is not
a price worth paying for it on CPU-only hardware. Revisit on the 250-item
synthetic set, where the power exists to tell +4 from 0.

**`recall@20` is identical in all three rows** — the reranker reorders stage
1's pool rather than retrieving anything new, so stage 1's recall@20 is the
hard ceiling on this entire stage.

**A reranker can make things worse.** `minilm-l6` improves recall@1 by +3.1
*and degrades recall@10 by −2.8*. Reordering a fixed pool is zero-sum:
promoting one document demotes another, and a 22M-parameter model is confident
enough to push a relevant chunk past rank 10. Its value therefore depends on
how many documents the generator actually reads — which couples D6 to the
Phase 5 context budget.

### ANN index — D4 · [full report](reports/04a_ann_lab.md)

5,796 chunks, 200 probe queries. Recall here is measured against **exact
search**, not human labels — it answers *what does approximation cost*.

| Index | Recall@10 | p50 | Memory |
|---|---:|---:|---:|
| **flat** ✅ | 1.0000 | 1.046 ms | 17.8 MB |
| hnsw | 1.0000 | **0.138 ms** | 19.4 MB |
| ivf | 0.9885 | 0.105 ms | 18.3 MB |
| ivfpq | 0.7995 | 0.561 ms | **1.7 MB** |

**A prior that failed.** We predicted flat would win outright at this scale. It
didn't — HNSW is *lossless* here and 7.6× faster. Flat is still chosen, for a
different reason than predicted: the 0.9 ms saving is under 0.1% of end-to-end
latency, and flat contributes **zero approximation error** while it serves as
the measurement instrument for every other Phase 4 experiment. Revisit at
~100k–250k chunks.

`ivfpq` is rejected on two counts — lossy *and* slower than exact search. Its
64 sub-quantizers each need 256 centroids from 5,796 vectors (~23 points per
centroid against a recommended ≥39), so the codebooks are undertrained. PQ is
a technique for tens of millions of vectors.

### Retrieval quality by question type

Where the system is currently weakest, on the golden set:

| Type | n | Recall@10 |
|---|---:|---:|
| multi_hop | 4 | 0.375 |
| comparison | 5 | 0.400 |
| synthesis | 5 | 0.467 |
| factual | 6 | 0.667 |
| explanation | 10 | 0.900 |
| definition | 5 | 1.000 |
| specific_source | 5 | 1.000 |

The three worst types are precisely those needing evidence from **multiple
passages**. That is the measured case for the multi-hop control flow in Phase
6 — not an assumption.

### Generation (Phase 5)

| Model | Faithfulness | Answer Relevancy | tok/s | RAM |
|---|---|---|---|---|
| *pending* | — | — | — | — |

### RAG control-flow ablation (Phase 6)

| Configuration | Correctness | Faithfulness | LLM calls | Latency |
|---|---|---|---|---|
| *pending* | — | — | — | — |

---

## Architecture

```
                    ┌──────────────┐
   query ──────────▶│    ROUTER    │  Adaptive RAG
                    └──┬────┬───┬──┘
       no_retrieval    │    │   │   multi_hop
            ┌──────────┘    │   └──────────┐
            │          simple│              ▼
            │               ▼           DECOMPOSE
            │      ┌────────────────┐      │
            │      │ QUERY TRANSFORM│◀─────┘
            │      │ rewrite/HyDE/  │
            │      │ expansion      │
            │      └───────┬────────┘
            │              ▼
            │      ┌───────────────────────────┐
            │      │  HYBRID RETRIEVE          │
            │      │  dense (FAISS/Chroma)     │
            │      │       +  BM25             │
            │      │       ↓  RRF fusion       │
            │      └───────┬───────────────────┘
            │              ▼
            │         RERANK (cross-encoder)
            │              ▼
            │         GRADE DOCS ─── CRAG
            │           ╱      ╲
            │       ok ╱        ╲ bad ──▶ rewrite ──┐ (cycle)
            │         ▼                             │
            │      GENERATE ◀───────────────────────┘
            │         ▼
            │      GROUNDEDNESS CHECK ─── Self-RAG
            │         ▼
            └──────▶ ANSWER + citations
```

**Key design choice:** CRAG, Self-RAG, Adaptive RAG and multi-hop are *not* four separate
pipelines. They are control flow over one shared retrieval core — which is why this uses
**LangGraph** (cycles, conditional edges) rather than linear LangChain chains. Each is a
node that can be toggled off, so "which advanced RAG method is best?" is answered by an
**ablation study**, not by four parallel implementations.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | *pending Phase 1* | PyMuPDF / pdfplumber / pypdf benchmarked on extraction quality |
| Chunking | *pending Phase 4* | recursive / semantic / parent-child |
| Embedding | *pending Phase 4* | nomic / bge-m3 / embeddinggemma / mxbai |
| ANN experiments | FAISS | only library exposing Flat/HNSW/IVF/PQ on identical vectors |
| Serving store | Chroma | persistence, metadata filtering, incremental adds |
| Lexical | BM25 | exact-term matching that embeddings blur |
| Orchestration | LangGraph | advanced RAG needs cycles |
| Generation | Ollama (local + cloud) | 3–4B local, `gpt-oss` cloud |
| Evaluation | DeepEval + GEval + LangSmith | component → RAG triad → application |
| API / UI | FastAPI + Streamlit | eval harness hits the same endpoint as the UI |

---

## Evaluation design

Three levels, each answering a different question:

**1. Component** — *is the retriever or the generator at fault?*
Contextual recall/precision/relevancy for retrieval; faithfulness and answer relevancy for
generation. Requires ground-truth **contexts**, not just answers — without them a failure
cannot be attributed.

**2. RAG Triad** — *is the pipeline internally consistent?*

```
            Question
           ╱        ╲
 Contextual          Answer
 Relevancy           Relevance
         ╲          ╱
  Context ──────── Answer
        Faithfulness
```

**3. Application** — *is it good, safe and operable?*
Quality (correctness, completeness, explanation style via GEval) · Safety (toxicity, PII
leakage, scope adherence) · Operations (latency p50/p95, token cost, error rate).

Every run is written to `evaluation/results.db`, keyed by config hash + git SHA, so
regressions stay visible across the project's history.

### Two methodological commitments

- **Ground truth is labelled by `(book, page_range)`, never by chunk ID.** Chunk IDs change
  with every chunking strategy; page ranges do not. This keeps one golden set valid across
  every experiment.
- **The judge is never the model being judged.** LLM judges exhibit self-preference bias —
  they score outputs resembling their own higher, and are blind to shared failure modes. Any
  same-family row is flagged and excluded from headline rankings.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
pip install -r requirements/base.txt       # phases 0-3; heavier layers added per phase
```

Place the PDFs in `data/raw/`. Then:

```bash
python -m src.ingest.parse_bench           # Phase 1: parser comparison
```

Requires [Ollama](https://ollama.com) for local models. Cloud models need `ollama signin`
and a **`:cloud`** tag suffix.

---

## Repository layout

```
configs/      experiment.yaml (the switchboard) + models.yaml (role → model registry)
src/          ingest · embed · index · retrieve · transform · graph · guardrails · serve
evaluation/   component · triad · application + results.db regression store
reports/      numbered findings — one per phase, each justifying a config value
data/golden/  the hand-labelled benchmark (committed; the corpus is not)
```

`configs/experiment.yaml` tags every value `[UNTESTED]`, `[PRIOR]` or `[PROVEN]`. A reviewer
can open that one file and see precisely which decisions have been earned.
