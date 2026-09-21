# Project Plan & Decision Log

The README reports the results. This file records **how the project was built**: the phases,
each architectural decision and the evidence behind it, and the lessons that shaped later work.

---

## Goal

Two deliverables, in order:

1. **The evidence.** Every architectural choice (parser, chunker, embedder, index, retrieval
   mode, reranker, control flow, generator) is a value in `configs/experiment.yaml`, justified
   by a numbered report in `reports/`.
2. **The product.** The winning configuration served as a working chatbot (FastAPI + Streamlit)
   with cited answers, conversation memory and guardrails.

Corpus: six ML/NLP textbooks: Géron, Goodfellow et al., Jurafsky & Martin, Bishop, Huyen,
Raschka (~3,500 pages).

---

## Phases

| # | Phase | Outcome | Report |
|---|---|---|---|
| 0 | Foundation | Config-driven layout, model registry, hardware budget | — |
| 1 | PDF parsing | `pymupdf` chosen from 4 parsers on a pre-registered rubric | [01b](reports/01b_parser_decision.md) |
| 2 | Ingestion | TOC-aware chunking, page-offset citations, incremental manifest | — |
| 3 | Benchmark data | Golden set (50, hand-curated) + synthetic set (250) | — |
| 4 | Retrieval lab | Chunker, embedder, index, retrieval mode and reranker decided | [04a](reports/04a_ann_lab.md) · [04b](reports/04b_embedder_bakeoff_hybrid_rrf_synthetic.md) · [04c](reports/04c_hybrid_retrieval.md) · [04d](reports/04d_reranking.md) · [04e](reports/04e_chunker_bakeoff_synthetic.md) |
| 5 | Generation | Generator and context budget decided | [05](reports/05_generators.md) · [05b](reports/05b_context_budget.md) |
| 6 | Control flow | Memory adopted; decomposition and CRAG measured and rejected | [06_memory](reports/06_memory.md) · [06_crag](reports/06_crag.md) |
| 7 | Guardrails | Injection and scope guards, calibrated on data | [07](reports/07_guardrails.md) |
| 8 | Evaluation | Faithfulness judge validated against human labels | [08](reports/08_judge_validation.md) |
| 9 | Serving | FastAPI + Streamlit; end-to-end acceptance test 7/7 | `scripts/acceptance_test.py` |

A phase counted as complete only with working code, a report containing measured numbers,
and a decision recorded below.

---

## Decision log

| # | Decision | Chosen | Evidence |
|---|---|---|---|
| D1 | PDF parser | **`pymupdf`** | Best on the rubric; pdfplumber merged words, which no metric caught but human review did ([01b](reports/01b_parser_decision.md)) |
| D2 | Chunking | **`recursive`**, 512 tokens / 64 overlap | Recursive, semantic and parent-child showed no significant difference (239–243 of 250 tied); chosen on build cost and bounded chunk size. Parent-child also tested at generation: no benefit ([04e](reports/04e_chunker_bakeoff_synthetic.md), [05b](reports/05b_context_budget.md)) |
| D3 | Embedder | **`embeddinggemma`** | +4 MRR over nomic and mxbai under hybrid (p ≤ 0.004); tied bge-m3, cheaper on every axis; ranking unchanged by quantisation ([04b](reports/04b_embedder_bakeoff_hybrid_rrf_synthetic.md)) |
| D4 | Vector index | **flat (exact)** | HNSW was lossless and 7.6× faster, but the saving is ~1 ms; exact search adds no approximation error to the other experiments ([04a](reports/04a_ann_lab.md)) |
| D5 | Retrieval mode | **`hybrid_rrf`** | +5.3 MRR over dense (p < 0.0001). Its recall gain disappeared once the embedder was strong ([04c](reports/04c_hybrid_retrieval.md)) |
| D6 | Reranker | **none** | +4.2 recall@1 within the noise band, at 1–6 s per query on CPU ([04d](reports/04d_reranking.md)) |
| D7 | Query transformation | **history-aware condensation** | Turn-2 recall on pronoun follow-ups 0.083 → 0.917. Query decomposition rejected: three variants, all below baseline ([06_memory](reports/06_memory.md), [06_decompose3b_keep](reports/06_decompose3b_keep.md)) |
| D8 | Control flow | **retrieve → generate + memory** | CRAG had no effect on recall (0W / 0L / 48T); its grader returned false negatives ([06_crag](reports/06_crag.md)) |
| D9 | Generator | **`gpt-oss:20b-cloud`** | Faithfulness 0.90 on human labels (llama3.2:3b 0.50); citations attached to claims in 36/36 answers; 6.6 s p50 ([05](reports/05_generators.md)) |
| D10 | Judge | **`gpt-oss:120b-cloud`** | Validated on 28 blind human labels: κ 0.815 on the production generator's answers, no self-preference ([08](reports/08_judge_validation.md)) |
| — | Context budget | **top 10, ≤ 5,300 tokens** | Wrong refusals 5 → 2 against top 6, faithfulness unchanged ([05b](reports/05b_context_budget.md)) |
| — | Guardrails | **patterns + corpus-similarity scope** | Injection 10/10 caught, 0 false positives on 300 real questions; scope blocks 0/50 golden questions ([07](reports/07_guardrails.md)) |

### Infrastructure choices

| Decision | Chosen | Why |
|---|---|---|
| Vector search | NumPy exact search (FAISS for the index benchmark) | 5.8k vectors search in ~1 ms; FAISS exposes Flat/HNSW/IVF/PQ on identical vectors |
| Orchestration | LangGraph | Advanced RAG patterns are cyclic; linear chains cannot express them |
| Memory | LangGraph SQLite checkpointer + condensation | Buffering history alone leaves follow-up questions unretrievable |
| Serving | FastAPI + Streamlit | UI and evaluation share one endpoint, so scores reflect what a user gets |
| Cloud tier | Ollama Cloud, `gpt-oss` family | The only free-tier cloud models |

---

## Lessons that shaped later work

**Measure before believing — including your own predictions.** Several expectations were
wrong once tested: HNSW turned out lossless, reranking was not the largest gain (hybrid fusion
was), query-prefix and quantisation effects were far smaller than assumed, and hybrid
retrieval's headline +11.8 recall came from the weaker initial embedder. Each correction is
recorded in the relevant report.

**Paired statistics, corrected for multiple comparisons.** Every configuration answers the same
questions, so the right test asks on how many questions A beat B. One apparent win for
weighted fusion (MRR, p = 0.03) did not survive Bonferroni correction across the three metrics
tested, and was rejected.

**The expensive bugs were silent.** Empty model answers from exhausted reasoning budgets,
results overwritten through filename collisions, cached responses reporting 0 ms latency, and
a local grader that exhausted RAM. None raised an error; each produced a plausible number.
The fixes (retry with a larger budget and temperature, per-dataset report paths, latency
stored with the cache, cloud grading) are in the code.

**Small local models are unreliable at judgement.** 1B and 3B models ignored the "leave simple
questions whole" rule 100% of the time when decomposing, and invented entities. Tasks that
decide what a turn is about (condensation, grading) therefore run on the cloud model.

**A judge must be validated, not assumed.** The faithfulness judge shares a model family
with the generator. Checked against human labels, it showed no self-preference; instead it
was lenient toward a *different* family's poorly placed citations.

---

## Hardware constraints

| Resource | Budget | Consequence |
|---|---|---|
| CPU | Ryzen 5 5600H, no usable GPU | Local generation capped at 3–4B; bulk embedding offloaded to a Colab T4 |
| RAM | 7.4 GB total | ~2 GB must stay free while serving; a local 3B grader could not coexist with the embedder |
| Cloud | Ollama free tier | Only `gpt-oss` models available for generation and judging |
