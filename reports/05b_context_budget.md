# Phase 5b — Context Budget and Parent-Child at Generation

Generator fixed at `gpt-oss:20b-cloud` (D9); retrieval fixed at the Phase 4 winner. One
variable per comparison. Judge `gpt-oss:120b-cloud` - SAME FAMILY as the generator, so
faithfulness figures carry that caveat until Phase 8's hand-label validation.

## Results

| config | distinct sources | context tok | usable | faithfulness | wrong refusals |
|---|---:|---:|---:|---:|---:|
| top_n=3  | 3  | 1,422 | 0.76 | **0.949** | 7 |
| parents  | 4  | 4,981 | 0.76 | 0.862 | 6 |
| top_n=6  | 6  | 2,713 | 0.76 | 0.873 | 5 |
| **top_n=10** | 10 | 4,537 | **0.80** | 0.871 | **2** |

**Nothing is significant after correction** (every usable pair p >= 0.75; the largest
faithfulness gap, top3 vs top6 +0.089, has a CI excluding zero but a sign test at
p=0.11). What follows is directional and is labelled as such.

## Finding 1 - a coverage/precision trade-off that cancels in the headline

Fewer chunks: the model cites the little it has precisely (faithfulness 0.949) but
cannot see the answer as often and refuses more (7). More chunks: fewer refusals (2),
slightly looser attribution (0.871). The two effects roughly cancel in `usable`, which
is why the headline barely moves across a 3x range of context.

## Finding 2 - refusals track BREADTH, not tokens

Parents (4,981 tokens) and top_n=10 (4,537 tokens) spend nearly the same budget, yet
parents refuse 6 times to top_n=10's 2. Wrong refusals fall monotonically with the
number of DISTINCT sources - 7, 6, 5, 2 for 3, 4, 6, 10 sources - regardless of how
many tokens those sources hold. Four whole sections cover fewer topics than ten small
chunks. Coverage comes from breadth.

## Finding 3 - parent-child does not help at generation either (closes D2)

D2 chose `recursive` on cost under a retrieval tie, and recorded that retrieval metrics
could not test parent-child's actual premise: *retrieve small, feed large*. This tests
it at a matched budget:

    usable  parents - chunks  -0.040  CI[-0.160,+0.080]  4W/6L/40T   p=0.7539
    faith   parents - chunks  -0.012  CI[-0.099,+0.070]  7W/6L/19T   p=1.0000

No benefit, slightly worse on every point estimate, and it pays for that with 3x the
wrong refusals. **D2 is now closed at both stages.**

## Decision

**Context: `top_n = 10`, budget ~5,300 tokens.** Chosen on direction, not significance:

- It directly attacks gpt-oss's main weakness from D9 - wrong refusals, 5 -> 2 -
  at no measurable faithfulness cost vs top_n=6 (0.871 vs 0.873, p=0.55).
- The mechanism (breadth reduces refusals) replicates across all four configs,
  including the parent-child run that was not designed to test it.
- The cost is prefill tokens, which on a cloud generator is marginal. It would NOT be
  marginal on a local CPU generator - this decision is coupled to D9.

**Coupling to D6 (reranker), noted.** The reranker reorders the top-20 and the generator
now reads the top 10. `minilm-l6` was measured DEGRADING recall@10 by 2.8 points, so at
top_n=10 the rejection of the reranker stands, arguably more firmly than before.

**Recorded:** `configs/experiment.yaml` generation.top_n / context_budget_tokens;
`plan.md` -> Phase 5 complete, D2 closed.
