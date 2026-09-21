# Phase 8 — Judge validation (gpt-oss:120b vs human)

28 citations labelled blind. The judge is the same family as the production generator (gpt-oss:20b), so the test is for LENIENCY toward its own family.

| generator | n | exact agreement | Cohen's κ | over-credit rate* |
|---|---:|---:|---:|---:|
| gpt-oss:20b-cloud | 15 | 0.93 | 0.815 | 0.00 (3 cases) |
| llama3.2:3b | 13 | 0.38 | 0.096 | 0.56 (9 cases) |

\* P(judge says *supported* | human said partial or unsupported).

## Verdict

**No large self-preference.** Over-credit on gpt-oss (0.00) is not materially above llama (0.56). At this n a SMALL bias cannot be ruled out, but not one large enough to reverse D9 - which was also corroborated judge-free (citation placement: 0/36 dumped vs 37/48).

## Interpretation

**The bias runs the other way.** The judge was not lenient toward its own family - it was
lenient toward the OTHER one. It over-credited 5 of 9 llama citations the human judged
partial or unsupported, and 0 of 3 gpt-oss ones.

Faithfulness recomputed from the human labels (supported = 1, partial = 0.5):

| generator | judge (Phase 5) | human labels |
|---|---:|---:|
| gpt-oss:20b-cloud | 0.87 | **0.90** |
| llama3.2:3b | 0.70 | **0.50** |

**D9 is strengthened, not weakened.** The judge understated the gap between the two.

**Why agreement collapses on llama (κ 0.10 vs 0.82).** llama prepends every citation
("[S1][S2][S3] The regularization term...") instead of attaching each to its claim, so
"does [S2] support the claim attached to it?" has no well-defined answer - human and
judge are each guessing which claim was meant. The judge is reliable on well-placed
citations, which is what the production generator produces (36/36 attached).

**Consequence for any future use of this judge:** trust it on gpt-oss:20b output (the
production path). Do not trust its faithfulness scores for a generator that dumps
citations - validate against human labels first.

**Limits.** n=28 citations. A small bias cannot be excluded; one large enough to reverse
D9 is excluded, and D9 was independently corroborated without any judge (citation
placement).
