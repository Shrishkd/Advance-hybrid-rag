# Advanced Hybrid RAG Chatbot

A retrieval-augmented chatbot over six machine-learning textbooks — built so that **every
architectural decision is measured rather than assumed**.

Most RAG projects pick a chunker, an embedder and a vector store on the first day and never
revisit them. This one treats each of those as an open question, benchmarks the alternatives
against a hand-labelled golden dataset, and ships the winner. The tables below are the
project; the chatbot is what falls out of them.

> **Status:** complete. Phases 0–9 done; Phase 10 (fine-tuning) deferred with reasons. Every served setting traces to a report in
> [`reports/`](reports). Live board: [`plan.md`](plan.md).

## Run it

```bash
powershell -File scripts/start_ollama.ps1                 # models live on D:
uvicorn src.serve.api:app --port 8000                     # the RAG API
streamlit run app/streamlit_app.py                        # the chat UI
python scripts/acceptance_test.py                         # end-to-end check: 7/7
```

Needs ~2 GB free RAM. On a 7.4 GB machine, close other heavy apps first: with two IDEs
and a browser open, free RAM fell to 0.2 GB and every model call stalled.

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

### Every decision, and what earned it

| | Decision | Evidence |
|---|---|---|
| D1 parser | `pymupdf` | pdfplumber collapsed word boundaries — **no metric caught it**; the human veto did |
| D2 chunker | `recursive` | all 3 strategies measured, **none significant** (239–243/250 ties); chosen on build cost |
| D3 embedder | `embeddinggemma` | +4 MRR over nomic/mxbai under hybrid (p≤0.004); ties bge-m3, wins every cost axis |
| D4 index | `flat` | HNSW is lossless *and* 7.6× faster — flat kept as the zero-error measurement instrument |
| D5 retrieval | `hybrid_rrf` | **+5.3 MRR, p<0.0001** vs dense |
| D6 reranker | **none** | +4.2 recall@1 for 1–6 s/query; inside the noise band |
| D9 generator | `gpt-oss:20b-cloud` | faithfulness 0.87 vs llama3.2:3b 0.70; 6.6 s vs 61 s |
| context | top 10, ~5.3k tokens | wrong refusals 5 → 2 |
| graph | **memory** only | turn-2 recall 0.083 → 0.917; decomposition rejected |

Significance throughout is **paired** (bootstrap + sign test, Bonferroni-corrected) —
every config answers the same questions, so "on how many did A beat B" is the test.

### Retrieval mode — D5 · [full report](reports/04c_hybrid_retrieval.md)

| Config (embeddinggemma, n=250) | Recall@10 | MRR |
|---|---:|---:|
| **hybrid_rrf** ✅ | 0.964 | **0.833** |
| hybrid_weighted | 0.964 | 0.847 |
| dense | 0.944 | 0.780 |
| bm25 | 0.920 | 0.777 |

**A headline that did not survive.** Measured first with the weaker `nomic` embedder,
hybrid beat dense by **+11.8 recall@10**. Re-run on the D3 winner, the recall gain
**vanishes** (p=0.18, 241/250 ties). What survives is ranking: hybrid beats *both* of
its own arms on MRR by ~5 points (p<0.0001). Fusion stops finding more once dense is
good — it keeps ordering better.

**BM25 alone beat dense** under nomic: technical textbooks are full of tokens that must
match literally (`ReLU`, `AdaGrad`, `CKY`), and dense embeddings place `Adam` near
`RMSProp`. RRF vs weighted is a genuine tie (twice failed significance); RRF wins on
having no free parameter.

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

### Generation (Phase 5) · [report](reports/05_generators.md)

| Model | usable | faithfulness | cites dumped up front | p50 latency |
|---|---:|---:|---:|---:|
| **gpt-oss:20b-cloud** ✅ | 0.76 | **0.87**¹ | **0/36** | **6.6 s** |
| llama3.2:3b | **1.00** | 0.70 | 37/48 | 61 s |
| phi4-mini | 0.68 | 0.55 | 19/35 | 73 s |
| qwen3:4b | **0.00** | — | — | ~330 s |

Each measurement layer changed who looked best. `grounded_rate` rated qwen3:4b perfect —
it was 14 truncated reasoning monologues that "cited" `[S1]`. `usable` rated llama3.2:3b
perfect — until the faithfulness judge found 29% of its answers carried an unsupported
citation, and a regex (no judge) showed why: it prepends every source tag instead of
attaching each to its claim. ¹ Judged by gpt-oss:120b, the same family. **Validated
against blind human labels** ([report](reports/08_judge_validation.md)): no
self-preference (κ 0.815 on gpt-oss answers); the judge was instead *lenient toward
llama*. By human labels the gap is wider: **gpt-oss 0.90, llama3.2:3b 0.50**.

### RAG control flow (Phase 6)

| Configuration | recall@10 | multi-passage recall | usable |
|---|---:|---:|---:|
| **baseline** (retrieve → generate) | **0.854** | **0.714** | **0.80** |
| + decompose (llama3.2:1b) | 0.726 | 0.488 | 0.74 |
| + decompose (llama3.2:3b) | 0.722 | — | 0.76 |
| + decompose (3b) + keep original | 0.774 | 0.655 | 0.72 |
| + CRAG (grade top 5, rewrite once) | 0.854 | 0.714 | 0.76 |

**CRAG had no effect** (0W/0L/48T on recall): its grader marked all five passages
irrelevant on 18 questions whose retrieval was already perfect — probably because it sees
only the first 400 characters of each chunk. Fusing the rewrite with the original query,
rather than replacing it, is why those needless rewrites cost latency but no recall.

**Decomposition is a negative result.** Both small models split 49/50 questions,
ignoring the rule to leave single-topic ones whole; the 1B also swapped topics ("Markov
blanket" → "Markov chain") and invented entities.

| Memory | turn-2 recall@10 on pronoun follow-ups |
|---|---:|
| off | 0.083 |
| **on** (history-aware condensation) | **0.917** |

### Guardrails (Phase 7) · [report](reports/07_guardrails.md)

Deterministic by design. Injection: **10/10 caught, 0 false positives** on 300 real
questions. Scope (similarity to the corpus, threshold calibrated on data): blocks **0/50**
golden and 2/250 held-out questions, catches 15/20 off-topic prompts — the rest are
declined by the answer prompt's `NOT_IN_SOURCES` rule.

## Skipped or deferred — and why

| Item | Status | Why |
|---|---|---|
| **Decomposition / multi-hop** | rejected | 3 configs, all below baseline (above) |
| **CRAG** | built, not enabled | no gain (above); one config line turns it on |
| **Adaptive router** | deferred | depends on a small model deciding "does this question need X?" — the exact judgement both small models failed 100% of the time for decomposition |
| **Self-RAG answer grading** | deferred | its cheap form is already live: `citations.py` verifies every `[S<n>]` and the API flags fabrications |
| **Reranker** | rejected | +4.2 recall@1 inside the noise band, 1–6 s/query on CPU |
| **HyDE / query expansion** | skipped | small models drifted when rewriting; not worth a third rewrite experiment |
| **Toxicity guard** | gap | `llama-guard3:1b` failed to download (IPv6); not substituted |
| **PII via Presidio** | substituted | regex for email/phone/card; misses names and addresses |
| **Token streaming** | deferred | answers arrive whole (~7 s); UI shows a spinner |
| **LangSmith tracing** | optional | needs an API key; the graph `trace` field shows each turn's path |
| **Fine-tuning (Phase 10)** | deferred | retrieval recall@10 is already 0.964 on the synthetic set; the plan says fine-tune only if justified |
| **gemma3:4b** | unavailable | pull failed (IPv6); three other local generators compared |

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
.venv\Scripts\activate
pip install -r requirements/base.txt -r requirements/retrieval.txt
pip install -r requirements/graph.txt -r requirements/serve.txt
python -m src.ingest.pipeline                 # parse + chunk data/raw/*.pdf
python -m evaluation.retrieval_lab --embed-only   # or the Colab notebook
```

Cloud models need `ollama signin` and a **`:cloud`** tag suffix. Only the `gpt-oss`
family is on the free tier.

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
