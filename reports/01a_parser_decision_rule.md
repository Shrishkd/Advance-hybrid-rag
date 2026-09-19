# Phase 1 — Parser Selection Rule (PRE-REGISTERED)

> **Written before the benchmark was run.** The point of pre-registration is that a rule
> chosen after seeing the data is not a rule, it is a rationalisation. If this rule turns out
> to select a parser that the excerpts show is bad, we change the rule *and say so in the
> report*, rather than quietly adjusting weights until the preferred parser wins.

**Decision being made:** D1 in `plan.md` — which parser `corpus.parser` is set to in
`configs/experiment.yaml`.

**Candidates:** `pymupdf`, `pymupdf_sorted`, `pdfplumber`, `pypdf`

---

## 0. The constraint that shapes everything

We have **no ground-truth transcription** of these 3,500 pages. Therefore *extraction
accuracy is not computable*. Any absolute threshold like "alpha_ratio must exceed 0.8" would
be invented, not derived.

Consequences for how thresholds are set:

- **Absolute thresholds are used only where a value is catastrophic *by definition*** —
  zero text, an exception, or output that is no longer recognisably prose. These are
  derived from properties of English text, **not** from our measurements.
- **Everything else is scored relatively**, parser-vs-parser on the *identical page*. This
  is sound because the page is held constant: if A recovers 2,400 characters and B recovers
  900 from the same page, B lost content, and we know that without ground truth.
- **A human veto overrides the arithmetic**, because proxy metrics are gameable.

---

## 1. The metrics

Five metrics, three roles. Not all of them select; some only diagnose.

| # | Metric | Direction | Role |
|---|---|---|---|
| 1 | `chars_per_page` | higher better | gate + discriminator |
| 2 | `alpha_ratio` | higher better | gate + discriminator |
| 3 | `single_char_token_ratio` | lower better | discriminator |
| 4 | `broken_hyphen_rate` | lower better | discriminator |
| 5 | `repeated_line_ratio` | lower better | **diagnostic only — not scored** |

### Why `repeated_line_ratio` does not score

Running headers and footers are **removed by the Phase 2 cleaner regardless of which parser
we choose**. Penalising a parser for faithfully extracting a header we intend to delete
would select for parsers that lose content at page edges — the opposite of what we want.

It is still reported, because a parser scoring ~0.00 where others score ~0.10 is probably
not "clean", it is probably dropping page furniture *and* whatever else lives near the
margins. Low is suspicious, not good.

---

## 2. Tier 1 — Hard gates (absolute; fail = disqualified)

| Gate | Threshold | Derivation |
|---|---|---|
| **G1 Errors** | 0 extraction errors on sampled pages | A parser that throws on our corpus is unusable at any quality. |
| **G2 Content recovery** | `chars_per_page` ≥ **0.90 ×** the best parser *on that same page* | Differences under 10% are explainable by legitimate handling differences (whitespace collapsing, ligatures, hyphen joining). Above 10% means whole blocks are absent. The 10% tolerance is a judgment call and is stated as such. |
| **G3 Still text** | corpus-mean `alpha_ratio` ≥ **0.55** | Derived from English, not from our data: ordinary prose runs ≈0.78 alphabetic among non-whitespace characters; even heavily mathematical pages stay ≈0.60–0.70. 0.55 is a floor for *"this is still text"*, not a quality bar. |

G2 is the gate that catches the failure mode proxies cannot otherwise see: **a parser that
silently drops every equation block scores *better* on metrics 2, 3 and 4 while destroying
the most content.** Only content recovery exposes it.

---

## 3. Tier 2 — Weighted score (ranks the survivors)

Metrics are on different scales and directions, so each is **min–max normalised per page**
across the surviving parsers, direction-corrected so 1.0 = best on that page:

```
DEAD-BAND CHECK FIRST:
    if (max - min) / max < 0.02:
        all parsers score 1.0 on this metric for this page
        -> the metric did not meaningfully discriminate; record as a tie

otherwise:
    higher-is-better:  s = (v - min) / (max - min)
    lower-is-better:   s = (max - v) / (max - min)
```

### AMENDMENT 1 (2026-09-19) — added after partial data, rule NOT otherwise changed

The original text said only *"if max == min, all parsers score 1.0"*. Partial measurement
(metrics 1–2, all 6 books) showed that exact equality is not the case that matters —
**near**-equality is:

| Parser | chars/page | alpha_ratio |
|---|---|---|
| pymupdf | 1639 | 0.914 |
| pymupdf_sorted | 1638 | 0.915 |
| pdfplumber | 1650 | 0.913 |
| pypdf | 1639 | 0.914 |

Spread: **0.7%** on chars, **0.2%** on alpha. Under plain min–max those differences
normalise to a full 0.0–1.0 range, so a 0.7% measurement difference would carry the same
weight in the final score as a catastrophic one. That is manufacturing signal from noise.

The 2% dead band is set below the ~3% sampling-noise estimate already used for the Tier-6
tie-break, so it is consistent with the rest of the rule rather than tuned to this data.

**Nothing else was altered.** Weights, gates, veto and tie-break stand as originally
written. Recording the amendment rather than silently editing, because a pre-registration
that gets quietly revised is worth nothing.

Then a weighted mean across metrics, then a plain mean across pages.

| Metric | Weight | Justification for this weight |
|---|---|---|
| `single_char_token_ratio` | **0.40** | Equation shatter is the dominant failure mode for *this* corpus. Two of six books (Bishop, Goodfellow) are equation-dense, and a third (Jurafsky) is formula-heavy. This is the metric most predictive of whether the corpus survives. |
| `broken_hyphen_rate` | **0.30** | Destroys vocabulary permanently and unrecoverably at retrieval time; affects all six books uniformly rather than a subset. |
| `chars_per_page` | **0.20** | Content recovery beyond the G2 gate — rewards recovering *more*, not merely *enough*. |
| `alpha_ratio` | **0.10** | Weakest signal: correlates with 3 and 4, and is confounded by legitimate mathematical notation, which is content we want. |

**Sensitivity check (required):** also compute a rank-based version (rank parsers 1–4 per
metric per page, weighted mean of ranks). If the rank ordering and the normalised ordering
disagree on the winner, the result is unstable — investigate before deciding, and report it.

---

## 4. Tier 3 — Human veto (mandatory, overrides Tier 2)

Read raw excerpts from **at least 6 pages spanning all 6 books**, including **≥2
equation-dense pages** (Bishop and/or Goodfellow) and **≥1 multi-column page**.

Automatic disqualification regardless of score:

- **Interleaved columns** — reading order broken across a two-column layout
- **Headers glued mid-sentence** into body text
- **Systematic ligature loss** (`ﬁ` → dropped, `ne-tuning` for `fine-tuning`)
- **Whole missing blocks** visible against another parser's output on the same page

Rationale: every Tier-2 metric is a proxy. Proxies can be satisfied by output that a human
can see at a glance is wrong. The arithmetic ranks; the eyes decide.

---

## 5. Tier 4 — Capability requirement

The chosen parser must support **embedded TOC extraction**, or be explicitly paired with one
that does.

Measured already (`plan.md` log): `pymupdf` and `pypdf` recover identical TOC counts on all
six books. **`pdfplumber` exposes no outline API at all.** So if `pdfplumber` wins Tier 2 we
must either adopt a two-parser pipeline (pdfplumber for text, pymupdf for structure) or take
the runner-up. That added complexity is a real cost and counts against it.

---

## 6. Final selection algorithm

```
1. Apply Tier-1 gates            -> survivors
2. If survivors == 0             -> escalate to out-of-scope candidates
                                    (Docling, Marker) and re-run
3. Rank survivors by Tier-2 weighted score
4. Apply Tier-3 veto to the top-ranked
     vetoed -> drop it, take the next, repeat
5. Apply Tier-4 to the winner
     no TOC support -> pair with pymupdf, OR take runner-up
                       if within 5% on the weighted score
6. TIE-BREAK: if the top two are within 3% of each other,
   prefer the FASTER parser.
```

### Why a 3% tie-break band

The sample is **8 pages × 6 books = 48 pages per parser**. At that size a 3% difference in a
normalised mean is not distinguishable from sampling noise. Declaring a winner inside that
band would be false precision.

Speed is the tiebreak rather than a scored metric because it compounds: ~3,500 pages
re-ingested on every chunking-strategy experiment in Phase 4, repeatedly. A parser 10× slower
converts a 1-minute ingest into 10 minutes, every iteration, forever.

---

## 7. What gets recorded

On completion:

- `reports/01_parsing_quality.md` — the tables, the excerpts, and the **Decision** section
- `plan.md` — decision log row **D1**, with the evidence link
- `configs/experiment.yaml` — `corpus.parser` set, tag upgraded `[PRIOR]` → `[PROVEN]`
- **If the rule was overridden** (e.g. Tier-3 veto changed the outcome), say so explicitly
  and explain why. An overridden rule that is documented remains honest; a silently adjusted
  one does not.

---

## 8. Stated prior (to be falsified)

Claude's expectation before measurement: **`pymupdf_sorted` wins**, on the grounds that
several books are multi-column and `sort=True` fixes reading order, and that PyMuPDF is
fast enough to re-ingest casually.

Recording it here so the benchmark can contradict it. If `pypdf` — the naive tutorial default
— turns out to be competitive, that is a publishable finding and it ships in the README.
