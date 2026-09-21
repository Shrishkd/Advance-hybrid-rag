# plan.md — Living Roadmap & Decision Log

> This file is **not** a copy of the README. The README reports conclusions.
> This file tracks what is still unknown, and records each decision **at the moment
> it becomes evidence-backed**.

---

## Goal

Two deliverables, in order:

1. **The evidence.** Every architectural choice — chunker, embedder, ANN index, retrieval
   mode, reranker, RAG control flow, generator — is a value in `configs/experiment.yaml`
   justified by a numbered report in `reports/`.
2. **The product.** The winning config is frozen into `configs/production.yaml` and shipped
   as a working chatbot (FastAPI + Streamlit), with citations, memory and guardrails.

Corpus: 6 ML/NLP textbooks (Géron, Goodfellow, Jurafsky, Bishop, Chip Huyen, Raschka),
plus a personal GenAI notes PDF added later via incremental ingestion.

---

## Status board

| Phase | Name | Status | Report |
|---|---|---|---|
| 0 | Foundation & guardrails | ✅ done | — |
| 1 | PDF parsing quality harness | ✅ **done** — D1 = `pymupdf` | [`01b_parser_decision.md`](reports/01b_parser_decision.md) |
| 2 | Ingestion + incremental manifest | ✅ **done** — 5,796 recursive / 8,944 parent-child chunks | — |
| 3 | Golden dataset + synthetic set | ✅ **done** — `golden_50.jsonl` (50, 0 errors) + `synthetic_retrieval.jsonl` (250, book-balanced) | — |
| 4 | Retrieval lab | ✅ **done** — D2 ✅ D3 ✅ D4 ✅ D5 ✅ · D6 measured, not adopted | [04a](reports/04a_ann_lab.md) · [04b](reports/04b_embedder_bakeoff_hybrid_rrf_synthetic.md) · [04c](reports/04c_hybrid_retrieval.md) · [04d](reports/04d_reranking.md) · [04e](reports/04e_chunker_bakeoff_synthetic.md) |
| 5 | Generation & model comparison | ✅ **done** — D9 `gpt-oss:20b-cloud`; top_n=10; parent-child closed | [05](reports/05_generators.md) · [05b](reports/05b_context_budget.md) |
| 6 | LangGraph: transforms, memory, advanced RAG | ✅ **done** — memory adopted; decomposition and CRAG measured and rejected; adaptive/Self-RAG deferred (README) | [06_memory](reports/06_memory.md) |
| 7 | Guardrails | ✅ **done** — injection 10/10 caught, 0 false positives on 300 real questions; scope blocks 0/50 golden, 2/250 held-out, catches 15/20 off-topic; toxicity NOT covered (llama-guard unavailable) | [07](reports/07_guardrails.md) |
| 8 | Evaluation | ✅ **done** — served config: usable 0.80, recall@10 0.854, faithfulness 0.87 (judge) / 0.90 (human); judge validated | [08](reports/08_judge_validation.md) |
| 9 | Ship the chatbot | ✅ **done** — FastAPI + Streamlit; acceptance test 7/7 | `scripts/acceptance_test.py` |
| 10 | Fine-tuning (Colab/Kaggle) | ⏸ **deferred** — retrieval recall@10 already 0.964; plan says only if justified | — |

**Rule: a phase is done only when it has working code, a report containing real numbers,
a row in the decision log below, and a passed comprehension check.**

---

## Decision log

Nothing enters this table on intuition. `Evidence` must name a report.

| # | Decision | Chosen | Status | Evidence |
|---|---|---|---|---|
| D1 | PDF parser | **`pymupdf`** | ✅ **PROVEN** | [`01b_parser_decision.md`](reports/01b_parser_decision.md) |
| D2 | Chunking strategy | **`recursive`** — all 3 measured, none significant | ✅ **CLOSED** — chosen on cost; parent_child tested at generation in Phase 5, no benefit | [`04e_chunker_bakeoff_synthetic.md`](reports/04e_chunker_bakeoff_synthetic.md) |
| D3 | Embedding model | **`embeddinggemma`** | ✅ **PROVEN** | [`04b_…hybrid_rrf_synthetic.md`](reports/04b_embedder_bakeoff_hybrid_rrf_synthetic.md) |
| D4 | ANN index type | **`flat`** (`IndexFlatIP`) | ✅ **PROVEN** | [`04a_ann_lab.md`](reports/04a_ann_lab.md) |
| D5 | Retrieval mode (dense/BM25/hybrid) | **`hybrid_rrf`** (+5.3 MRR, p<0.0001) | ✅ **PROVEN** — re-run on `embeddinggemma`; recall gain vanished, ranking gain held | [`04c_hybrid_retrieval.md`](reports/04c_hybrid_retrieval.md) |
| D6 | Reranker | `minilm-l12` on quality, **not adopted** | 🟡 provisional — gain +4.2 r@1 is inside noise; cost 3–6 s/query (contended) | [`04d_reranking.md`](reports/04d_reranking.md) |
| D7 | Query transform (rewrite/expansion/HyDE) | *pending* | ⬜ Phase 6 | — |
| D8 | RAG control flow | **memory only** — decompose ❌ (3 runs < baseline) · CRAG ❌ (no gain, grader false negatives) | ✅ **decided** | [06_memory](reports/06_memory.md) · [06_crag](reports/06_crag.md) |
| D9 | Generator model | **`gpt-oss:20b-cloud`** — faith 0.87, 6.6 s | ✅ **decided by Shrish** — same-family judge ⇒ Phase 8 judge validation REQUIRED | [`05_generators.md`](reports/05_generators.md) |
| D10 | Judge model | `gpt-oss:120b-cloud` | ✅ **validated** — κ 0.815 on gpt-oss answers, no self-preference (over-credit 0.00); UNRELIABLE on citation-dumping generators (κ 0.10) | [08](reports/08_judge_validation.md) |

### Decisions already fixed (infrastructure, not experiments)

These were chosen for reasons that don't require benchmarking:

| Decision | Chosen | Why |
|---|---|---|
| Vector store | FAISS to experiment, Chroma to serve | FAISS is the only one that exposes Flat/HNSW/IVF/PQ on identical vectors, which *is* experiment D4. Chroma handles persistence, metadata filters and incremental adds. |
| Orchestration | LangGraph | CRAG/Self-RAG/Adaptive need cycles and conditional edges. Chains are linear and cannot express them. |
| Frontend | FastAPI + Streamlit | The eval harness hits the same served endpoint as the UI, so scores reflect what a user actually gets. |
| Memory | LangGraph `SqliteSaver` + query condensation | `ConversationBufferMemory` is deprecated; and buffering alone leaves follow-up questions unretrievable. |
| Cloud tier | Ollama free tier, `gpt-oss` only | Verified: everything else 402s on this account. |

---

## Open questions

**Two models dropped as unavailable, 2026-09-20.** `gemma3:4b` and `llama-guard3:1b` both
failed to pull after repeated retries — TLS handshake timeout and connection reset to
Cloudflare **over IPv6**. Not disk, not config: other pulls the same day succeeded, and
38 GB was free. Shrish authorised dropping both.

- **`gemma3:4b`** cost: one row in the D9 table. Three local generators of the same size
  class remain plus a cloud model, and D9 turns on local-vs-cloud rather than on which 4B
  model wins. No follow-up needed.
- **`llama-guard3:1b`** cost: Phase 7 loses its purpose-built safety classifier.
  `plan.md` listed it as a *candidate*, never a locked decision. Substituting prompted
  `llama3.2:1b` (already installed, sized for classification).

**Why the substitution is defensible, and where it is weak.** Llama Guard screens
general chat harms — self-harm, violence, hate — while this system answers questions
about ML textbooks. The guardrails that carry weight here are scope adherence,
groundedness and PII, and **groundedness is already solved without any model** by
`src/generate/citations.py`, which proves every `[S<n>]` resolves to a supplied source.
A prompted classifier is also inspectable and tunable where Llama Guard is a black box.

**But this is a substitution, not a finding.** It has NOT been benchmarked against the
specialist and cannot be until the pull succeeds. Phase 7's report must state that
plainly rather than presenting the substitute as a chosen design. If either model ever
pulls on a different network, both become one more thing to measure — nothing in the
project breaks meanwhile.


**Colab handoff is built and ready** (`notebooks/embed_colab.ipynb`). Produces all four
embedders' vectors on a T4 in minutes instead of ~6 h locally. Backend is part of the
vector cache key: Ollama serves quantised GGUF, Colab runs fp16, and the same checkpoint
gives different vectors — so a D3 run mixing the two would compare *quantisation* as much
as *models*. `corpus_cache` raises rather than computing a non-ollama vector locally, and
`import_colab_vectors.py` refuses anything failing a fingerprint, row-count, normalisation
or duplicate-model check. **Note this breaks the CLAUDE.md "corpus never leaves the
machine" rule, under explicit instruction**; the export ships text and a line index only —
book, page, chapter and chunk_id stay local, so ground-truth labels never leave.

**Semantic chunking is now wired** (Phase 4 unblocked the embedder dependency) but
**unrun**. It embeds every sentence — ~90k against 5.8k chunks, roughly 15× a full corpus
pass, several hours on this CPU. It is the most expensive preprocessing step in the
project and is a strong candidate for the Colab path too. D2 is incomplete until it runs.

**D2/D3 circularity, resolved by fiat:** semantic chunking needs an embedder to produce
chunks, and D3 compares embedders over chunks. We pin the config's current
`embedding.model` for chunk *production* and record it, so the dependency is visible
rather than hidden.



1. **OpenAI embeddings** — originally requested, no API key available. Implemented behind the
   embedder interface (`configs/models.yaml` → `openai-3-small`, `enabled: false`). One line
   to switch on; ~$0.05 for the full corpus.
2. **`bge-m3` sparse + ColBERT vectors** — Ollama serves dense only. True multi-vector hybrid
   needs `FlagEmbedding`, likely on Colab. Decide in Phase 4 whether it's worth the complexity.
2b. ~~**Goodfellow has no embedded TOC**~~ — **RESOLVED 2026-09-19.** Shrish sourced a
   different edition of the same book (`deeplearningbook.pdf`, 800pp) that ships a proper
   outline: **194 entries, depth 3**, breadcrumbs resolving correctly
   (`Part III: Deep Learning Research > 14 Autoencoders > 14.9 Applications`).
   **All six books now have embedded TOCs; no heading-detection fallback is needed.**
   Worth remembering as a general lesson: sourcing a better input beat writing a parser
   workaround. Minor residue — the publisher's own outline contains typos (`Treands`,
   `ModernPractices`, `Recurrentand`), so never exact-string-match on section titles.
3. **GenAI notes PDF** — arrives later. Phase 2's content-hash manifest makes this a no-op.
4. **Reranker latency on CPU** — `bge-reranker-v2-m3` (568M) may be too slow to serve on this
   box even if it wins on quality. If so, the honest outcome is a documented
   quality-vs-latency tradeoff, not a hidden compromise.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Free-tier rate limits during Phase 8 | LLM response cache keyed by `(model, prompt_hash, params)`. Never pay twice for an identical prompt. |
| `gpt-oss` reasoning traces leak into scored answers | `strip_reasoning: true`; assert in tests that no output contains `Thinking...`. |
| Self-preference bias on the `gpt-oss:20b` generator row | Flag the row, exclude from headline ranking, cross-check against hand labels. |
| n=50 golden set is statistically underpowered | Judge-free retrieval metrics run on a ~250-item synthetic set where labels are cheap. |
| Self-correction loops running forever | `graph.max_retries` cycle guard. |
| 7.4 GB RAM exhaustion | Small batch sizes; state RAM cost before loading any model; 4B param ceiling. |

---

## Log

**2026-09-22 - Judge validated; the feared bias ran the other way.**

The Phase 8 obligation (validate a same-family judge against human labels) came back
clean for the production path: on gpt-oss:20b answers the judge agreed with Shrish's
blind labels 93% of the time (kappa 0.815) and never over-credited a citation. It was
LENIENT toward llama3.2:3b instead (over-credit 0.56), so by human labels gpt-oss 0.90 vs
llama 0.50 - D9's margin was understated, not overstated. Agreement on llama collapses
(kappa 0.10) because its dumped citations make "which claim does [S2] support?"
ill-defined for anyone. All phases 0-9 complete; Phase 10 deferred.


**2026-09-21 - Phase 7 done; one of my claims measured and corrected.**

Guardrails are deterministic by design, since Phase 6 showed small local models are
unreliable at yes/no judgement. Injection is pattern-matched (10/10 attacks, 0 false
positives on 300 real questions). Scope is similarity to the corpus, with the threshold
CALIBRATED as the lowest golden score minus 0.02 = 0.294: it blocks 0/50 golden, 2/250
held-out synthetic, and catches 15/20 off-topic prompts; the 5 that pass are declined
downstream by NOT_IN_SOURCES.

I justified running scope only on a session's first turn by claiming pronoun follow-ups
have "near-zero" corpus similarity. Measured: 0.265-0.453, mostly ABOVE the threshold.
The rule survives on different evidence - 2 of 15 follow-ups ("Go on.", "Really?") fall
below and several clear it by under 0.02, so a mid-conversation check would block
follow-ups at random. The docstring and report now carry the measured version.


**2026-09-21 - Decomposition rejected; memory adopted.**

Decomposition was the textbook fix for multi-passage questions. Three configurations,
all worse than the baseline (recall@10 0.854):

    decompose, llama3.2:1b                 0.726
    decompose, llama3.2:3b                 0.722
    decompose, llama3.2:3b + keep original 0.774

The failure is the GATE, not the split. Both small models split 49/50 and 8/8 questions
regardless of the prompt's "leave single-topic questions whole" rule, and fragments of
a single-topic question retrieve worse than the whole. The 1B model also swapped topics
("Markov blanket" -> "Markov chain") and invented entities (Huyen's lending study became
"UC Berkeley admissions"). A better model improved the splits and changed nothing in
the result. Keeping the original query recovered ~5 points but the original is one vote
among four in RRF, and weighting it would be a free parameter fitted to 50 questions.

Memory (history-aware condensation, gpt-oss) took turn-2 recall on pronoun-only
follow-ups from 0.083 to 0.917 over 12 conversations - the Phase 9 acceptance criterion.
Condensation uses the cloud generator, not a small local model, on the evidence above:
it is the one rewrite that decides what the whole turn is about.


**2026-09-21 - Phase 6 baseline locked; porting it found three silent client bugs.**

The baseline graph (retrieve -> generate) was required to reproduce Phase 5 exactly
before any node is added, or every Phase 6 delta would be confounded by the port:
retrieval 50/50 identical ranked lists, prompt context 50/50 identical, replay 50/50 in
9 s. Getting there exposed three bugs in `src/llm/client.py`, none of which raised:

1. **Retried answers were not cached under the original key.** The first call came back
   empty, so it was never cached, so it went LIVE on every replay - and a hosted model is
   not deterministic across calls even at temperature 0. 5 of 50 answers (10%) changed
   between two runs of an identical config. Phase 6 measures few-point effects; a 10%
   noise floor would have made the ablation meaningless.
2. **Degenerate loops skipped the retry entirely.** The retry required non-empty
   reasoning. gpt-oss has a mode where it burns the whole budget (`done_reason=length`)
   and returns NEITHER reasoning NOR content, so those came back silently empty.
3. **More budget does not fix a loop.** At 8,192 tokens, 3/3 degenerate prompts still
   returned nothing after 100+ s each. At temperature 0.8 all 3 terminated normally;
   0.4 did not escape. The retry now raises budget AND temperature (fixed seed 7), capped
   at 4,096 tokens so a loop fails fast rather than slowly.

**Retroactive note on D9.** gpt-oss:20b-cloud's `usable` in `05_generators.md`
(0.76) and `05b` (0.76-0.80) were depressed by 4-8% empty answers now recovered. The
chosen model is somewhat BETTER than measured. D9 stands - it won anyway - and the
tables are left as measured rather than silently revised.


**2026-09-21 00:40 — PAUSED mid-D9. Resume point recorded.**

Phase 5's generator comparison was stopped part-way to free the machine. **Nothing is
lost**: every generation is cached in `data/processed/llm_cache.sqlite` (5.3 MB, on
disk), so re-running the exact command re-uses completed work and only does what remains.

    gpt-oss:20b-cloud   50/50  done
    llama3.2:3b         50/50  done
    qwen3:4b            14/50  partial
    phi4-mini            0/50  not started

**Resume with the identical command** — the model list must match, because the cache is
keyed on (model, prompt, params) and a different prompt or top_n is a different key:

    python -m evaluation.generation_lab       --models gpt-oss:20b-cloud,llama3.2:3b,qwen3:4b,phi4-mini

**Finding already banked: qwen3:4b is ~6x slower than llama3.2:3b** — 5.5 min/question
vs 53 s, on 4B vs 3B params, which parameter count does not explain. `models.yaml`
calls it "hybrid reasoning mode", and gpt-oss was already measured IGNORING
`think=False`; the same is almost certainly happening here. At 5.5 min/question it may
be disqualified on latency before quality is scored, since Phase 9 ships an interactive
UI. Worth confirming by inspecting its reasoning trace length rather than assuming.

Caveat on those timings: local models were timed while contending with each other, which
inflates all of them. The cloud model (~9 s/question) was measured under the same
conditions, so the local-vs-cloud gap is if anything understated.


**2026-09-20 - D3 decided, and a second claim of mine measured and found wrong.**

`embeddinggemma` wins D3. The interesting part is not the winner:

**Hybrid fusion almost erases the embedder's effect on recall.** Dense-only,
embeddinggemma beat mxbai by 6.0 recall@10 points with 15 wins and 0 losses across
250 questions. Add BM25 and all six pairs go statistically indistinguishable, 244-247
ties out of 250, with bge-m3 vs embeddinggemma at literally +0.0000. The spread across
four embedders falls from 0.060 to 0.012.

But MRR does NOT converge: embeddinggemma still beats nomic (p=0.0043) and mxbai
(p=0.0002) under hybrid. Recall asks *did we find it*, MRR asks *how high did it land*.
Fusion answers the first for us and leaves the second open. Since the generator reads
the top few chunks, the second is closer to what a user experiences.

**The quantisation cross-check falsified my own claim.** I had written that mixing
Ollama-quantised and Colab-fp16 vectors would cause "a systematic quantisation error
that varies by model", and built machinery to prevent it. Measured: cosine between the
two versions of the same chunk is 0.99999 mean / 0.9965 worst, and recall@10 across 250
questions is IDENTICAL to four decimals for all four models. Only MRR moves, in the
fourth decimal.

So the guard was right and the reason given for it was wrong. The separation cost
nothing and the error's size was unknown until measured - but I stated a magnitude I had
not checked, for the second time this project. The docstring now carries the numbers.

The useful finding that fell out: **quantisation is free for retrieval at this scale.**
Serving through Ollama gives up nothing measurable, so fp16 serving is off the list of
things worth engineering.

**Also fixed: the st run silently overwrote the ollama report and per-question files.**
Neither path carried the backend. Same bug class as the dataset collision fixed earlier -
two valid runs, one filename.


**2026-09-20 - The noise band I quoted all project was the wrong statistic.**

I repeatedly said "at n=48 a 5-point gap is noise." That is the standard error of an
UNPAIRED proportion. Our comparisons are PAIRED - every config answers the same
questions - so the correct question is not "could two samples of 48 differ this much"
but "on how many individual questions did A beat B."

Built `evaluation/significance.py` (paired bootstrap + sign test) and re-examined the
D5 tie I had called by eye:

    hybrid_rrf vs hybrid_weighted, recall@10, n=48
      diff +0.0069, 95% CI [-0.0347, +0.0486], sign test 3W/2L/43T, p=1.0000

43 ties in 48. The call was right; it is now measured. This required persisting
PER-QUESTION scores - an aggregate is lossy, and once you hold only the mean you can
never ask how many questions each config actually won.

**2026-09-20 - I nearly flipped D5 on a p-hacked result.**

On the synthetic 250, `hybrid_weighted` led every headline number and looked like
grounds to revisit D5. Tested across three metrics, exactly one cleared p<0.05 (MRR,
p=0.031). Three metrics were tested; Bonferroni puts the threshold at 0.0167 and it does
not survive.

Two further reasons it would have been wrong even if it had:

1. **245 of 250 ties on recall@10.** Whatever separates the methods lives entirely in
   ordering within a near-identical result set.
2. **The synthetic set's known bias points exactly that way.** Its questions were
   written while looking at the passage, which flatters lexical retrieval - and weighted
   fusion is the method that amplifies a confident lexical arm, since it blends raw
   scores rather than ranks. A win there is what the bias predicts, not evidence against
   RRF.

The tool now takes `--n-comparisons` and says plainly when a result fails correction,
because "run three tests, report the one that passed" is a mistake that is easy to make
honestly.

**2026-09-20 - D6: the second prior to fail.**

`experiment.yaml` predicted reranking would be "the biggest single quality win per unit
effort." Measured: hybrid fusion gave +11.8 recall@10 for +40 ms; the best reranker gave
+4.2 recall@1 for +3,000 ms. Reranking is the SECOND-biggest win and by far the most
expensive thing in the project. **Not adopted.**

The finding worth keeping is that `minilm-l6` improves recall@1 by +3.1 while DEGRADING
recall@10 by -2.8. Reordering a fixed pool is zero-sum: promoting one document demotes
another, and a 22M-parameter cross-encoder is confident enough to push a genuinely
relevant chunk past rank 10. So a reranker's value depends on how many documents the
generator reads - which couples D6 to the Phase 5 context budget, and means the two
cannot be decided independently.

`recall@20` is identical across all three rerankers by construction, which keeps the real
ceiling visible: a reranker cannot recover what stage 1 never retrieved.


**2026-09-19 - D5: a prior confirmed, and an assumption I did not know I held.**

The pre-registered prior said hybrid would win because textbooks are full of exact
terms. It won by +11.8 recall@10 - the first effect in this project comfortably
outside the noise band.

What I had not predicted: **BM25 alone beats dense** (0.7326 vs 0.7049), at half the
latency and with a 1.3-second index build. I had quietly assumed the neural embedder
was the strong arm and BM25 the supporting one. The table says otherwise. Fusing them
still beats either, so the conclusion is "complementary", not "BM25 wins" - but the
ordering surprised me and that is worth recording.

The mechanism was visible in a number I nearly skipped past: dense has recall@5 ==
recall@10 exactly. Ranks 6-10 contribute *nothing*. Dense finds the passage in its top
5 or never finds it at all. BM25 shows the mirror image - worse at k=5, better at k=10.
Good coverage, noisy ordering versus good ordering, capped coverage.

**2026-09-19 - D4: my prior was wrong, and the decision stayed the same anyway.**

I predicted flat would win outright at ~6k chunks. HNSW turned out to be *lossless*
(recall@10 = 1.0000) and 7.6x faster. Flat is still chosen, but for a reason I had not
articulated in advance: it is the **measurement instrument** for every other Phase 4
experiment, and an approximate index would mix its own error into every embedder and
chunker comparison. The right answer for the wrong reason is still a wrong prediction.

**2026-09-19 - Four silent-wrong bugs in one session, all in the measurement layer.**

None of these raised an exception. All of them would have produced a plausible report.

1. `is_cached()` called `np.lib.format._read_array_header`, **removed in numpy 2.x**.
   The AttributeError landed in a bare `except` and the function returned False
   forever. Effect: the "cached" fast path never ran and the log lied about what was
   being recomputed.
2. `write_report()` selected table columns from `results[0]`. In the rerank sweep row 0
   is `none`, which has no `rerank_ms_p50` - so the reranker's latency would have
   vanished from the report while it rendered perfectly. Fixed to use the union of keys.
3. **Decision sections were being destroyed by reruns.** The ANN lab was hardened
   against this; the retrieval lab was not, and a rerun deleted a written D5 decision.
   The fix had lived in one script instead of a shared function. Now
   `evaluation/report_utils.py`, imported by both.
4. `precision@5` looked alarming at 0.2458 until the ceiling was computed: the labels
   cap it at 0.7125, because 13 of 48 items have fewer than 5 relevant chunks in
   existence. The measured value is 34% of achievable, not 25% correct. Raw precision
   without its ceiling is close to meaningless, so the ceiling now ships in the table.

The pattern from Phase 1-2 continues to hold: **the dangerous bugs in a RAG pipeline
are the ones that return a number.**

**2026-09-19 - gpt-oss ignores `think=False`.**

Synthetic question generation returned empty strings on 6/6 attempts at
`max_tokens=200`: the model emitted ~950 characters of reasoning and had no budget left
for the answer. `think=False` is passed and does not suppress it. The only lever is a
budget that covers reasoning *plus* output - now 800. The warning added to
`src/llm/client.py` in Phase 0 is what caught this immediately instead of leaving 250
blank questions.


**2026-09-19 — A claim of mine failed its own test.**

I stated, in `plan.md` and again in `src/embed/base.py`, that getting query/document
prefixes wrong costs **10-20% recall**. I then measured it: 60 corpus chunks, nomic,
section titles as queries.

| | recall@1 | recall@5 | MRR |
|---|---|---|---|
| with prefixes | 0.550 | 0.817 | 0.673 |
| without prefixes | 0.567 | 0.767 | 0.671 |

Recall@1 is marginally *worse* with prefixes; MRR is identical to three decimals. At n=60
that is noise, and nothing like 10-20%.

Two honest caveats: the probe used section TITLES as queries, which are short and
keyword-like rather than the natural-question-vs-passage asymmetry prefixes are trained
for — so it is a weak proxy that settles nothing. And prefixes stay applied regardless,
because they are the model's documented interface and cost nothing.

But the number was repeated as established fact before being measured, inside a docstring
arguing against exactly that habit. Claim softened in code; the magnitude question is now
an open item for `reports/04_retrieval_lab.md`, to be answered with the golden set's real
questions.


**2026-09-19 — Two incremental-ingestion bugs, both from the same root cause.**

The manifest tracks **documents**; chunks are produced **per strategy**. Conflating those
broke incremental ingestion twice, in different ways:

1. *First half (already patched):* switching to a strategy whose chunks did not exist
   reported `unchanged` and produced an empty corpus. Patched with a chunk-file existence
   check.
2. *Second half (missed):* after a `PIPELINE_VERSION` bump, running `pipeline` then
   `pipeline --strategy parent_child` reprocessed **only the first**. The recursive run
   stamped the *document* as current, so the parent_child run saw `stale=0 unchanged=6`
   and skipped — leaving v2 chunks on disk under a manifest asserting they were v3.
   The existence check could not catch it, because the files existed; they were just wrong.

Fixed properly with `DocRecord.strategy_versions` so freshness is tracked per strategy.
*Lesson: patching the symptom (missing files) left the general defect (wrong granularity)
in place. The second failure was the same bug wearing different clothes.*

Also fixed: physical citations now render **1-based** ("PDF p. 1" for the first sheet).
`cite_page` was emitting the raw 0-based index, so 172 chunks cited "PDF p. 0" — correct
internally, confusing to anyone verifying against a PDF reader that labels it page 1.
Page ranges in metadata stay 0-based, so golden labels are unaffected.

**2026-09-19 — Phase 0 started.**
Probed environment rather than assuming it. Findings that changed the plan:
- Ollama cloud tags **require** a `:cloud` suffix (`glm-5.3` silently resolves to a
  non-existent local tag).
- Only `gpt-oss:20b-cloud` / `gpt-oss:120b-cloud` are free; glm / kimi / deepseek all 402.
  → Judge-vs-judge bake-off replaced with cloud-judge vs local-judge validation, which is
  better teaching anyway: it *demonstrates* why a small local judge can't be trusted.
- `gpt-oss` emits visible `Thinking...` traces → must be stripped before scoring.
- Ollama defaults to `C:` (31.6 GB free); redirected to `D:\ollama-models`.

**2026-09-19 — Phase 1 code complete. All 6 PDFs ingested (96 MB).**

*TOC recovery probe — the central bet of this phase, measured not assumed:*

| Book | Entries | Depth | Outcome |
|---|---|---|---|
| Speech & Language (Jurafsky) | 548 | 4 | ✅ Part > Ch > Sec > Sub |
| Pattern Recognition (Bishop) | 285 | 3 | ✅ |
| Hands-On ML (Géron) | 166 | 3 | ✅ |
| Build a LLM (Raschka) | 109 | 3 | ✅ |
| Designing ML Systems (Huyen) | 105 | 3 | ✅ |
| **Deep Learning (Goodfellow)** | **0** | — | ❌ **no outline** — see open question 2b |

`pymupdf` and `pypdf` recovered **identical** TOC counts on all six, so TOC extraction does
**not** discriminate between them — D1 will be decided on text quality alone. `pdfplumber`
exposes no outline API at all, which is itself disqualifying if it wins on text.

*Bug found and fixed in `SectionIndex`:* resolving each outline level independently produced
incoherent citations — Bishop p379 rendered as `8. Graphical Models > 7.2. Relevance Vector
Machines` because page 379 sits inside chapter 8 but before its first numbered section, so the
level-2 search returned chapter 7's last section. Both halves individually plausible, together
nonsense. Replaced with a backward walk that only accepts strictly-shallower ancestors.
Also generalised `Breadcrumb` from three fixed fields to a variable-depth path, because
Jurafsky nests four levels and would otherwise have had its *parts* mislabelled as chapters.

**2026-09-19 — Parser decision rule PRE-REGISTERED** (`reports/01a_parser_decision_rule.md`),
written before the benchmark, so the rule cannot be reverse-engineered from the result.

**2026-09-19 — Partial measurement (metrics 1–2 only, 4 pages × 6 books × 4 parsers).**

| Parser | chars/page | alpha_ratio | ms/page |
|---|---|---|---|
| pymupdf | 1639 | 0.914 | **64.5** |
| pymupdf_sorted | 1638 | 0.915 | 78.2 |
| pdfplumber | **1650** | 0.913 | 1232.1 |
| pypdf | 1639 | 0.914 | 556.8 |

Findings:
- **All four pass every Tier-1 gate** comfortably (alpha ~0.91 vs a 0.55 floor). The gates
  only catch catastrophic failure; they contribute nothing to *this* decision.
- **Metrics 1–2 do not discriminate** — 0.7% and 0.2% spread. These are born-digital
  publisher PDFs with clean text layers, so every library reads the same text stream.
  Triggered Amendment 1 (2% dead band) to stop min–max normalisation amplifying that
  noise to full scale.
- **Speed discriminates enormously**: pymupdf is **19× faster** than pdfplumber and
  **8.6× faster** than pypdf.
- **One exception worth chasing:** on Bishop (PRML), pdfplumber recovers **+2.8% more
  characters** (2242 vs 2180) — the only book where metric 1 clears the dead band. Either
  it is recovering more real equation content, or more shattered fragments.
  `single_char_token_ratio` is exactly the metric that tells those apart. This is now the
  sharpest open question in Phase 1.

**Implication:** ~70% of the Tier-2 weight now rests on the two unimplemented metrics.
If they also tie, the tie-break rule decides and pymupdf wins on speed.

**2026-09-19 — Phase 3: candidate generation + curation tooling.**

Built `evaluation/generate_candidates.py` (drafts questions FROM sampled chunks, so
ground-truth contexts come free), `curate.py` (review → validate → promote), and
`find_chunks.py` (keyword search for writing the hand-labelled types).

Findings:

1. **A non-empty breadcrumb is not enough to identify content.** The sampling dry-run
   drew Jurafsky's `BIBLIOGRAPHICAL AND HISTORICAL NOTES` — a bibliography has a perfectly
   good breadcrumb. Added `is_content_section()` in `toc.py` (bibliographies, exercises,
   indexes, contents). Pool dropped Jurafsky 1508→1170, Bishop 759→627. The same filter
   fixed `find_chunks`, where **contents pages matched every query** because they list
   every topic and explain none.

2. **Token budget silently killed exactly the hardest question types.** First run returned
   61/80, and the losses were not random: comparison 2/6, multi_hop 3/8, synthesis 1/6,
   while single-passage types mostly succeeded. Paired-passage prompts produce longer
   reference answers, which overran `max_tokens=500` mid-JSON. Raised to 1200, capped
   answers at 70 words, added truncation salvage → failures fell from ~25% to ~3%.
   *The lesson is the correlation, not the fix: a budget failure looked like random
   flakiness until the failures were grouped by type.*

3. **Substring search lies about acronyms.** `"LoRA"` returned 26 hits across all six
   books **including Bishop (2006)** — fifteen years before LoRA existed. The matches were
   `exp-LORA-tion`. Whole-word search gives 2 hits, both Raschka (2024). Added `-w`.
   A hit in a book that predates the concept is the cheapest sanity check available.

4. **Negative printed page numbers.** Bishop's offset is −19, so front-matter physical
   page 13 rendered as "p. −6". Arithmetically right, semantically absurd, and it would
   have appeared in real citations. Front matter carries no printed Arabic number, so
   below 1 we now fall back to citing the physical page. *Pending: re-ingest to apply.*

Verified-absent topics for unanswerable questions (0 matches across all six books):
RLHF, direct preference optimization, constitutional AI, chain-of-thought, flash
attention, diffusion model, Mamba. `data/golden/manual.jsonl` seeded with three worked,
verified examples — one cross-document, one ambiguous ("kernel": 94 hits in Bishop as a
similarity function, 22 in Géron including convolution kernels), one unanswerable.

**2026-09-19 — PHASE 2 COMPLETE.** Both chunk corpora built over all 6 books:

| Book | Pages | TOC | recursive | parent-child (children/parents) |
|---|---|---|---|---|
| Raschka | 299 | 109 | 506 | 622 / 133 |
| Goodfellow | 800 | 194 | 1,340 | 2,038 / 343 |
| Huyen | 339 | 105 | 309 | 494 / 104 |
| Géron | 279 | 166 | 343 | 528 / 143 |
| Bishop | 758 | 285 | 1,327 | 2,078 / 376 |
| Jurafsky | 1,044 | 548 | 1,971 | 3,184 / 626 |
| **total** | **3,519** | | **5,796** | **8,944 / 1,725** |

All chunks within budget (recursive max 512, children max 256). Citations verified
end to end, including the `cite_kind` split: Géron renders "p. 147", Huyen renders
"PDF p. 161" because it has no trustworthy printed-page offset.

Incremental ingestion verified both ways: a re-run reports
`unchanged=6 -> processing 0`, while `--strategy parent_child` correctly reprocesses
despite the manifest being current, because chunks are per-strategy and the manifest
tracks documents. That asymmetry was a real bug — without the chunk-file existence
check, switching strategy silently produced an empty corpus.

**LLM client built** (`src/llm/client.py`) with the two things the free tier forces:
a SQLite response cache (verified: 1,214 ms → 0 ms on repeat) and gpt-oss reasoning
stripping (verified: `Thinking...` removed from the answer, retained in `.reasoning`).
Also fixed a retry loop that retried `TypeError` four times with backoff — retrying a
programming error just hides it behind 15 seconds of delay.

**2026-09-19 — Phase 2: first full ingestion of all 6 books.** Three bugs found by running
on real data that no synthetic fixture would have produced:

1. **Off-by-one across the whole citation layer.** PDF outlines are 1-indexed;
   `extract_page()` is 0-indexed. Every `SectionIndex` query mixed the two. It would never
   have crashed, never moved a metric, and rendered perfectly-formatted citations that were
   all wrong by one page. Normalised every adapter to emit 0-based pages.
   *This is the most dangerous bug class in the project — silent, plausible, and only
   detectable by opening the actual book.*

2. **A corpus about LLMs breaks LLM tooling.** Raschka's book contains the literal string
   `<|endoftext|>` as subject matter. tiktoken raises on special-token strings by default,
   assuming an injection attempt. Correct for prompts, wrong for counting — fixed with
   `disallowed_special=()`. Flagged for Phase 7: the same string reaching a *generator*
   prompt is a real guardrails question, not a counting one.

3. **BPE token counts are not additive.** `count(A) + count(B) != count(A+B)`. The sliding
   window summed per-atom counts, then emitted `text[start:end]`, which re-includes the
   separators split() discarded and re-tokenises across boundaries. Result: **646-token
   chunks against a 512 limit.** Would have failed silently — `mxbai-embed-large` has a
   512-token context and truncates without error, so retrieval would just quietly degrade.
   Fixed by counting the real candidate text; added a loud regression guard in the pipeline.
   → `PIPELINE_VERSION` bumped v1→v2, which correctly marked all six documents stale and
   forced reprocessing. The versioning mechanism proved itself on its first real use.

**2026-09-19 — Phase 2 in progress: manifest + cleaning done, chunkers next.**

*Equation policy — decided by Shrish, not benchmarked:* `preserve_equations: true`.
On this corpus the mathematics IS the content; stripping it would make
"what is the closed-form solution for ridge regression?" unanswerable. Accepted cost:
fragmented equation text depresses `alpha_ratio` and inflates `single_char_token_ratio`
in affected chunks. Noise kept in exchange for not losing content.

*Page-offset detection — measured on 60 consecutive pages per book:*

| Book | Offset | Agreement | Coverage | Verdict |
|---|---|---|---|---|
| Raschka | −21 | 1.00 | 1.00 | ✅ printed |
| Goodfellow | −14 | 1.00 | 1.00 | ✅ printed |
| Bishop | −19 | 1.00 | 1.00 | ✅ printed |
| Géron | −1 | 1.00 | 1.00 | ✅ printed (hand-verified) |
| Huyen | — | 0.28 | 0.53 | ❌ cite physical |
| Jurafsky | — | 0.50 | 0.60 | ❌ cite physical |

Two algorithm fixes were needed to get there, both found by testing on real books:
1. **Split "confidence" into coverage vs agreement.** One number conflated *"this book
   prints no page numbers"* with *"the numbers we found disagree"* — opposite problems
   demanding opposite fixes (look harder vs. stop looking at the wrong thing).
2. **Collect ALL edge-line number candidates per page, not just the first.** Stopping at
   the first match let a chapter heading in the top edge cast a bogus vote and mask the
   real page number below it. Fixing this moved Goodfellow 0.78→1.00, Bishop 0.58→1.00,
   Géron 0.88→1.00. The signal exploited: a true page number yields the *same* offset on
   *every* page, while chapter/figure/equation numbers scatter.

Both remaining failures are genuine properties of the books, not detector bugs:
Huyen prints no page numbers in the extracted edges (footnote markers sit where they
would be), and **Jurafsky's draft paginates per chapter** (`DRAFT / 30 / Chapter`), so no
global offset exists. `PageOffset.trustworthy` correctly rejected both. A confidently
wrong offset would put every citation in a book off by a constant — worse than admitting
we cite PDF pages.

**2026-09-19 — PHASE 1 COMPLETE. D1 = `pymupdf`.** Full reasoning in
[`01b_parser_decision.md`](reports/01b_parser_decision.md). Three findings worth carrying
forward:

1. **The human veto did the real work.** `pdfplumber` collapses word boundaries corpus-wide
   (`forthecancertreatmentproblem`) and leaks `(cid:N)` glyph codes — and **not one of the
   five metrics detected it**. `single_char_token_ratio` was actively *improved* by the
   damage, because gluing words together reduces single-character tokens. A fully automated
   selection rule would have ranked it 3rd for the wrong reasons and could have picked it
   under different weights. → **Phase 2 must add a `word_integrity` metric** (alphabetic
   tokens >22 chars, or a dictionary hit rate). The current metric set has a blind spot.
2. **Prior falsified.** `pymupdf_sorted` scored highest (0.978 vs 0.968) but inside the 3%
   noise band, at +24% cost. Reading-order sorting had nothing to fix on born-digital
   publisher PDFs. Kept available as one config value.
3. **The naive default is quantifiably bad on maths.** `pypdf` hits
   `single_char_token_ratio = 0.48` on Goodfellow vs 0.17 for `pymupdf` — near half of all
   tokens are single characters. `PyPDFLoader` is the standard tutorial choice; this is what
   it costs on a technical corpus. Ships in the README.

*Goodfellow TOC fallback — feasibility confirmed, not assumed:* `Chapter N` yields 0 matches,
but `N.M Title` yields 25 matches across 62 sampled pages (`1.2 Historical Trends`,
`5.7.2 Support Vector Machines`, `7.12 Dropout`). Chapter numbers derive from the section
prefix. Needs a filter for figure captions (`3.1 The two parameters µ ∈R...` is Figure 3.1,
not a section). **Decision: keep the book.**
