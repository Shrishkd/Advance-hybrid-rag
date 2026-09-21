# Phase 6 — `crag`

Nodes: `['crag']` · generator `gpt-oss:20b-cloud` · transform `llama3.2:1b` · top_n=10 · n=50

| metric | value |
|---|---:|
| usable | 0.76 |
| recall@10 | 0.8542 |
| usable_multi | 0.6429 |
| recall@10_multi | 0.7143 |
| wrong_refusals | 2 |
| mean_docs | 20.0 |
| wall_s_p50 | 19.3 |
| errored | 0 |

## By question type (indicative at n=4-10 per cell)

| type | n | recall@10 | usable |
|---|---:|---:|---:|
| ambiguous | 2 | 0.500 | 1.00 |
| comparison * | 5 | 0.900 | 0.60 |
| definition | 5 | 1.000 | 0.80 |
| explanation | 10 | 0.900 | 0.80 |
| factual | 6 | 1.000 | 0.83 |
| multi_hop * | 4 | 0.500 | 0.50 |
| numerical | 6 | 0.833 | 0.67 |
| specific_source | 5 | 1.000 | 0.80 |
| synthesis * | 5 | 0.700 | 0.80 |
| unanswerable | 2 | - | 1.00 |

\* multi-passage types - the target of Phase 6.

## Decision - built, measured, NOT adopted

| | baseline | CRAG | paired |
|---|---:|---:|---|
| recall@10 | 0.854 | 0.854 | 0W/0L/48T - no effect |
| usable | 0.800 | 0.760 | 0W/2L, p=0.50 - slight harm, ns |
| rewrites triggered | - | 18/50 | all on questions with baseline recall 1.0 |

**No measurable benefit, extra latency (one grade call per turn, plus a rewrite in 36% of
turns). Not enabled in the served graph.** It remains available as a toggle
(`graph.nodes: [crag]` in configs/experiment.yaml).

**Why it did nothing:** the grader produced false negatives. 41 of 99 grade replies marked
all five passages irrelevant, e.g. `1: no 2: no 3: no 4: no 5: no`, on questions whose
retrieval was already perfect. Not a parse failure - a genuine wrong judgement. Probable
cause: each passage is truncated to its first 400 of ~2,000 characters, so the grader
often never sees the answer-bearing part.

**What went right:** the rewrite is FUSED with the original query rather than replacing
it, so 18 unnecessary rewrites cost latency but no recall - the design choice carried over
from the decomposition failures.

**Not pursued further** (per the project's priority rule: no indefinite optimisation of a
component that shows no gain). Widening the snippet is the obvious next experiment if CRAG
is ever revisited. Earlier, a per-passage local grader (llama3.2:3b) could not run at all:
it did not fit in RAM beside the embedder and API and froze the machine at 22/50.
