# Advanced Hybrid RAG Chatbot

A retrieval-augmented chatbot over six machine-learning textbooks — built so that **every
architectural decision is measured rather than assumed**.

Most RAG projects pick a chunker, an embedder and a vector store on the first day and never
revisit them. This one treats each of those as an open question, benchmarks the alternatives
against a hand-labelled golden dataset, and ships the winner. The tables below are the
project; the chatbot is what falls out of them.

> **Status:** Phase 0 of 10 — foundation. No results yet. Every table below is a placeholder
> until its phase produces numbers. See [`plan.md`](plan.md) for the live status board.

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

### Retrieval (Phase 4)

| Config | Recall@10 | Precision@10 | MRR | nDCG@10 | Latency p50 |
|---|---|---|---|---|---|
| *pending* | — | — | — | — | — |

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
